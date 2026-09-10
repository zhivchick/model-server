import sys
import asyncio
import re
import time
import logging
import json
import mlx.core as mx
from mlx_lm.generate import stream_generate
from response_formatters import build_streaming_chunk, build_multi_tool_streaming_chunk
from qwen_xml_parser import QwenXmlParser

from perf_tracker import perf_tracker
from anti_loop import anti_loop_engine

logger = logging.getLogger("mlx_lm_server.bridge")

def _log_stream_start(request_id: str, is_utility: bool, prompt_len: int):
    print("-" * 60)
    if is_utility:
        logger.info(f"[GPU LOCK] Utility request {request_id} acquired exclusive GPU access.")
        print(f"⚙️ [Utility Stream | Context: {prompt_len} tokens] ", end="")
    else:
        logger.info(f"[GPU LOCK] Main agent {request_id} acquired exclusive GPU access.")
        print(f"🖥️ [Agent Stream | Context: {prompt_len} tokens] ", end="")
    sys.stdout.flush()

def _parse_xml_arguments(full_text: str) -> dict:
    param_matches = re.findall(r'<parameter=([^>]+)>(.*?)(?:</parameter>|$)', full_text, re.DOTALL)
    args_dict = {}
    for p_name, p_val in param_matches:
        clean_val = p_val.replace("</parameter>", "").strip()
        args_dict[p_name.strip()] = int(clean_val) if clean_val.isdigit() else clean_val

    if not args_dict and "{" in full_text:
        try:
            json_match = re.search(r'(\{.*?\})', full_text, re.DOTALL)
            if json_match: args_dict = json.loads(json_match.group(1))
        except Exception: pass
    return args_dict


def sync_generation_worker(model, tokenizer, prompt_ids, max_tokens, request_id, has_tools, prefill_step_size, global_cache, queue, loop, model_name, prompt_tokens_len, server_lock):
    """Background worker executing inside Starlette run_in_threadpool context."""
    logger.debug(f"🚀 Entering sync_generation_worker for request: {request_id}")
    
    mx.set_default_device(mx.gpu)
    is_utility = not has_tools
    _log_stream_start(request_id, is_utility, prompt_tokens_len)
    
    current_prefill_speed = 0.0
    current_decode_speed = 0.0
    chunk_len = len(prompt_ids)
    
    try:
        full_response_text = ""
        tokens_count = 0
        parser = QwenXmlParser()
        
        prefill_start_time = time.perf_counter()
        
        with mx.StreamContext(mx.default_stream(mx.gpu)):
            logger.debug("Initializing mlx_lm.generate.stream_generate loop instance...")
            generator_instance = stream_generate(
                model, tokenizer, prompt=prompt_ids, max_tokens=max_tokens, 
                prompt_cache=global_cache, prefill_step_size=prefill_step_size
            )

            try:
                first_response = next(generator_instance)
                logger.debug("Prefill step evaluation completed! First token extracted.")
                
                prefill_time = time.perf_counter() - prefill_start_time
                token = first_response.text
                full_response_text += token
                tokens_count += 1
                
                current_prefill_speed = chunk_len / prefill_time if prefill_time > 0 else 0.0
                
                is_tool = parser.parse_chunk(token) if has_tools else False
                if not is_tool:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, content=token)), loop
                    )
            except StopIteration: return

            # 🎯 ФИКС: ЯВНО ЗАСЕКАЕМ ВРЕМЯ СТАРТА ДЕКОДИРОВАНИЯ
            generation_start_time = time.time()
            logger.debug("Entering main Decoding token stream loop...")

            # Флаги и буфер для потоковой изоляции сырого JSON
            json_accumulating = False
            json_buffer = ""

            for response in generator_instance:
                token = response.text
                full_response_text += token
                tokens_count += 1
                
                # Сценарий Qwen 3.5: нативный XML-парсер
                in_tool_call = parser.parse_chunk(token) if has_tools else False
                if in_tool_call:
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    continue

                # Сценарий Qwen 2.5: перехватываем JSON-команды в буфер накопления
                if has_tools and ("{" in token or "```json" in token or json_accumulating):
                    if not json_accumulating:
                        json_accumulating = True
                        sys.stdout.write("\n⚙️ [JSON INTERCEPT ACTIVE] ")
                    
                    json_buffer += token
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    continue

                # Обычный Markdown-текст рассуждений транслируем пользователю в реальном времени
                sys.stdout.write(token)
                sys.stdout.flush()
                
                asyncio.run_coroutine_threadsafe(
                    queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, content=token)), loop
                )

        print()
        generation_time = time.time() - generation_start_time
        current_decode_speed = tokens_count / generation_time if generation_time > 0 else 0.0

        
               # === НЕУБИВАЕМЫЙ ПОСИМВОЛЬНЫЙ МНОЖЕСТВЕННЫЙ JSON-ПЕРЕХВАТЧИК ===
        is_raw_json_tool = False
        json_tool_calls_list = []
        
        if has_tools and json_accumulating:
            
            # Ищем все честные границы вложенных JSON-объектов через счётчик скобок
            raw_objects = []
            brace_count = 0
            start_pos = -1
            
            for pos, char in enumerate(json_buffer):
                if char == "{":
                    if brace_count == 0:
                        start_pos = pos
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0 and start_pos != -1:
                        # Нашли честный, закрытый со всеми вложениями объект!
                        raw_objects.append(json_buffer[start_pos:pos+1])
                        start_pos = -1
                              
            for idx, obj_str in enumerate(raw_objects):
                clean_str = obj_str.strip()
                try:
                    parsed_json = json.loads(clean_str)
                    if "name" in parsed_json and "arguments" in parsed_json:
                        is_raw_json_tool = True
                        args_obj = parsed_json["arguments"]
                        
                        # OpenAI формат требует, чтобы arguments внутри JSON был СТРОКОЙ
                        if isinstance(args_obj, dict):
                            stringified_args = json.dumps(args_obj, ensure_ascii=False)
                        else:
                            stringified_args = str(args_obj)
                            
                        call_payload = {
                            "index": idx,
                            "id": f"call_json_{idx}_{int(time.time())}",
                            "type": "function",
                            "function": {
                                "name": parsed_json["name"].strip(),
                                "arguments": stringified_args
                            }
                        }
                        json_tool_calls_list.append(call_payload)

                except Exception as je:
                    print(f"⚙️ [DEBUG MULTI-JSON] ❌ Cбой парсинга объекта {idx}: {str(je)}")

        sys.stdout.flush()


        # МАРШРУТИЗАЦИЯ И ВЫЗОВ РАДАРА АНТИ-ПЕТЛИ
        if has_tools and (parser.in_tool_call or is_raw_json_tool):
            if is_raw_json_tool and len(json_tool_calls_list) > 0:
                # Извлекаем первый элемент списка по ИНДЕКСУ, а уже из него берем ["function"]
                first_call = json_tool_calls_list[0]["function"]
                tool_invocation_name = first_call["name"]
                try:
                    extracted_args = json.loads(first_call["arguments"])
                except Exception:
                    extracted_args = {"raw_arguments": first_call["arguments"]}
            else:
                extracted_args = _parse_xml_arguments(full_response_text)
                tool_invocation_name = parser.tool_name


            sys.stdout.flush()
            
            t_name, t_args = anti_loop_engine.evaluate_and_process(full_response_text, tool_invocation_name, extracted_args)
            
            # 🎯 Сценарий 1. Жесткий вылет по повторам
            if t_name is None:
                asyncio.run_coroutine_threadsafe(
                    queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, content=f"\n\n🛑 {t_args}\n", finish_reason="stop", prompt_len=prompt_tokens_len, completion_len=tokens_count)), loop
                )
            # 🎯 Сценарий 2. Штатный ход
            elif t_name == tool_invocation_name:
                if is_raw_json_tool:
                    raw_chunk = build_multi_tool_streaming_chunk(request_id=request_id, model_name=model_name, tool_calls_list=json_tool_calls_list, prompt_len=prompt_tokens_len, completion_len=tokens_count)
                    
                    asyncio.run_coroutine_threadsafe(
                        queue.put(raw_chunk), loop
                    )
                else:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, finish_reason="stop", prompt_len=prompt_tokens_len, completion_len=tokens_count)), loop
                    )
            # 🎯 Сценарий 3. Подмена ответа шелла
            else:
                asyncio.run_coroutine_threadsafe(
                    queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, tool_name=t_name, tool_args=t_args, finish_reason="tool_calls", prompt_len=prompt_tokens_len, completion_len=tokens_count)), loop
                )

        else:
            asyncio.run_coroutine_threadsafe(
                queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, finish_reason="stop", prompt_len=prompt_tokens_len, completion_len=tokens_count)), loop
            )
            

        perf_tracker.record_metrics(
            current_prefill_speed, current_decode_speed, 
            total_context_len=prompt_tokens_len, prompt_chunk_len=chunk_len, completion_len=tokens_count
        )

    except Exception as e:
        logger.exception(f"Critical exception inside GPU worker execution loop: {str(e)}")
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        logger.info(f"[GPU RELEASED] Request {request_id} execution finalized.")
        print("-" * 60)




async def async_queue_bridge(model, tokenizer, prompt_ids, max_tokens, request_id, has_tools, prefill_step_size, global_cache, model_name, prompt_tokens_len, server_lock):
    from starlette.concurrency import run_in_threadpool
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    asyncio.create_task(
        run_in_threadpool(
            sync_generation_worker, model, tokenizer, prompt_ids, max_tokens, request_id, has_tools, 
            prefill_step_size, global_cache, queue, loop, model_name, prompt_tokens_len, server_lock
        )
    )
    while True:
        chunk = await queue.get()
        if chunk is None: break
        yield chunk
    yield "data: [DONE]\n\n"
