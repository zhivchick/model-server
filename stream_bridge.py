import sys
import asyncio
import re
import time
import logging
import json
import threading
import mlx.core as mx
from mlx_lm.generate import stream_generate
from response_formatters import build_streaming_chunk
from qwen_xml_parser import QwenXmlParser

logger = logging.getLogger("mlx_lm_server.bridge")
GPU_THREAD_LOCK = threading.Lock()

def _log_stream_start(request_id: str, is_utility: bool):
    """Фиксирует захват GPU потоком."""
    if is_utility:
        logger.info(f"⚙️ [GPU LOCK] Фоновая утилита {request_id} зашла на монопольный обсчет.")
        print("⚙️ [Utility Stream] ", end="")
    else:
        logger.info(f"🔒 [GPU LOCK] Основной агент {request_id} захватил монопольный доступ к GPU.")
        print("🖥️ [Agent Stream] ", end="")
    sys.stdout.flush()

def _process_tool_call(full_text: str, parser: QwenXmlParser) -> tuple:
    """Извлекает имя инструмента и валидирует его аргументы через JSON."""
    tool_name = parser.tool_name if parser.tool_name else "execute"
    final_json_args = parser.extract_final_arguments()
    
    # Резервный фолбэк, если структура сломалась, но внутри есть сырой JSON
    if final_json_args == "{}" and "{" in full_text:
        try:
            json_match = re.search(r'(\{.*?\})', full_text, re.DOTALL)
            if json_match:
                final_json_args = json.dumps(json.loads(json_match.group(1)), ensure_ascii=False)
        except Exception:
            pass
            
    logger.info(f"Generated Tool Call: '{tool_name}' with args: {final_json_args}")
    return tool_name, final_json_args

def sync_generation_worker(model, tokenizer, prompt_ids, max_tokens, request_id, has_tools, prefill_step_size, global_cache, queue, loop, model_name, prompt_tokens_len):
    """Фоновый воркер с абсолютной изоляцией GPU-потоков и точной трансляцией токенов."""
    global GPU_THREAD_LOCK
    GPU_THREAD_LOCK.acquire()
    
    is_utility = not has_tools
    _log_stream_start(request_id, is_utility)
    
    try:
        full_response_text = ""
        tokens_count = 0
        parser = QwenXmlParser()
        
        generator_instance = stream_generate(
            model, tokenizer, 
            prompt=prompt_ids, 
            max_tokens=max_tokens, 
            prompt_cache=global_cache, 
            prefill_step_size=prefill_step_size
        )

        # ⚡️ Извлечение первого токена (Prefill стадия)
        try:
            first_response = next(generator_instance)
            token = first_response.text
            full_response_text += token
            tokens_count += 1
            
            is_tool = parser.parse_chunk(token) if has_tools else False
            if not is_tool:
                asyncio.run_coroutine_threadsafe(
                    queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, content=token)), 
                    loop
                )
        except StopIteration:
            return

        generation_start_time = time.time()

        # 🏃‍♂️ Потоковый вывод (Decoding стадия)
        for response in generator_instance:
            token = response.text
            full_response_text += token
            tokens_count += 1
            
            in_tool_call = parser.parse_chunk(token) if has_tools else False
            
            # Если пошел XML-вызов инструмента, скрываем сырые теги от UI Goose
            if in_tool_call:
                sys.stdout.write(token)
                sys.stdout.flush()
                continue

            sys.stdout.write(token)
            sys.stdout.flush()
            
            asyncio.run_coroutine_threadsafe(
                queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, content=token)), 
                loop
            )

        print("\n" + "-" * 50 if not is_utility else "")
        
        # 🎯 Простая математика контекста: берем переданную точную длину
        total_prompt_context_len = prompt_tokens_len

        # Формируем финальный чанк с метриками Usage для трекера Goose
        if has_tools and parser.in_tool_call:
            t_name, t_args = _process_tool_call(full_response_text, parser)
            asyncio.run_coroutine_threadsafe(
                queue.put(build_streaming_chunk(
                    request_id=request_id, model_name=model_name, 
                    tool_name=t_name, tool_args=t_args, finish_reason="tool_calls", 
                    prompt_len=total_prompt_context_len, completion_len=tokens_count
                )), loop
            )
        else:
            asyncio.run_coroutine_threadsafe(
                queue.put(build_streaming_chunk(
                    request_id=request_id, model_name=model_name, finish_reason="stop", 
                    prompt_len=total_prompt_context_len, completion_len=tokens_count
                )), loop
            )
            
        generation_time = time.time() - generation_start_time
        if not is_utility and generation_time > 0:
            logger.info(f"📊 [TRACKER DEBUG] Отправлен контекст: {total_prompt_context_len} токенов.")
            logger.info(f"Generated {tokens_count} tokens in {generation_time:.2f} s ({tokens_count / generation_time:.2f} t/s)")

    except Exception as e:
        logger.exception(f"Критический сбой внутри GPU-воркера: {str(e)}")
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        GPU_THREAD_LOCK.release()
        logger.info(f"🔓 [GPU RELEASED] Запрос {request_id} завершен, видеочип свободен.")

async def async_queue_bridge(model, tokenizer, prompt_ids, max_tokens, request_id, has_tools, prefill_step_size, global_cache, model_name, prompt_tokens_len):
    """Асинхронный транзитный шлюз между пулом потоков и SSE потоком FastAPI."""
    from starlette.concurrency import run_in_threadpool
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    asyncio.create_task(
        run_in_threadpool(
            sync_generation_worker, 
            model, tokenizer, prompt_ids, max_tokens, request_id, has_tools, 
            prefill_step_size, global_cache, queue, loop, model_name, prompt_tokens_len
        )
    )
    
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk
    yield "data: [DONE]\n\n"
