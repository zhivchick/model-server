# stream_bridge.py
import sys
import asyncio
import re
import time
import logging
import json
import mlx.core as mx
from mlx_lm.generate import stream_generate
from response_formatters import build_streaming_chunk

logger = logging.getLogger("mlx_lm_server.bridge")

def sync_generation_worker(model, tokenizer, prompt_ids, max_tokens, request_id, has_tools, prefill_step_size, global_cache, queue, loop, model_name, prompt_tokens_len):
    """Синхронный фоновый воркер генерации с честными таймингами и именованными вызовами чанков."""
    mem_before = mx.get_active_memory()
    
    current_cache_bytes = sum(c.v.nbytes if hasattr(c, "v") and c.v is not None else 0 for c in global_cache)
    logger.info(f"Prompt Cache active VRAM: {current_cache_bytes / 1e9:.4f} GB")
    logger.info(f"Evaluating tokens chunk: {len(prompt_ids)} tokens")
    print("-" * 50)
    
    is_thinking = False
    in_tool_call = False
    full_response_text = ""
    tokens_count = 0
    
    # Инициализируем генератор MLX
    generator_instance = stream_generate(
        model, tokenizer, 
        prompt=prompt_ids, 
        max_tokens=max_tokens, 
        prompt_cache=global_cache, 
        prefill_step_size=prefill_step_size
    )
    
    # 🎯 ЧЕСТНЫЙ МАТЕМАТИЧЕСКИЙ ЗАМЕР СКОРОСТИ PREFILL
    prefill_start = time.perf_counter()
    try:
        # Си-слой Apple Silicon блокирует поток и считает prefill
        first_response = next(generator_instance)
        prefill_end = time.perf_counter()
        
        prefill_time = prefill_end - prefill_start
        prefill_speed = prompt_tokens_len / prefill_time if prefill_time > 0 else 0
        
        logger.info(f"[{request_id}] Real Prefill Finished in {prefill_time:.4f} s ({prefill_speed:.2f} prompt tokens/s)")
        print("-" * 50)
        
        token = first_response.text
        full_response_text += token
        tokens_count += 1
        
        # Фикс: Именованная передача первого токена
        asyncio.run_coroutine_threadsafe(
            queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, content=token)), 
            loop
        )
    except StopIteration:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        return

    generation_start_time = time.time()

    # Основной цикл посимвольного декодинга (Phase 2: Decoding)
    for response in generator_instance:
        token = response.text
        full_response_text += token
        tokens_count += 1
        
        # Фильтр скрытых мыслей Qwen (<think>...</think>)
        if "<think>" in full_response_text and not is_thinking:
            is_thinking = True
            sys.stdout.write("🧠 [Thinking] ")
            sys.stdout.flush()
            continue
        if "</think>" in full_response_text and is_thinking:
            is_thinking = False
            full_response_text = full_response_text.split("</think>")[-1]
            print("\n🎯 [Thinking Finished]")
            continue
        if is_thinking:
            sys.stdout.write(".")
            sys.stdout.flush()
            continue

        # Перехват вызова инструментов
        if has_tools and ("<tool_call>" in full_response_text or "<function=" in full_response_text):
            if not in_tool_call:
                in_tool_call = True
                print("\n🛠️  [Model Called Tool] Перехватываем XML поток...")
            sys.stdout.write(token)
            sys.stdout.flush()
            continue

        # Обычный текстовый поток — выводим на экран
        sys.stdout.write(token)
        sys.stdout.flush()
        
        # Фикс: Именованная передача текстового чанка в Goose
        asyncio.run_coroutine_threadsafe(
            queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, content=token)), 
            loop
        )

    print("\n" + "-" * 50)
    
    # Пост-парсинг XML тегов в OpenAI JSON
    if in_tool_call:
        func_match = re.search(r'<function=([^>]+)>', full_response_text)
        tool_name = func_match.group(1).strip() if func_match else "tree"
        param_matches = re.findall(r'<parameter=([^>]+)>(.*?)(?:</parameter>|$)', full_response_text, re.DOTALL)
        args_dict = {}
        for p_name, p_val in param_matches:
            clean_val = p_val.replace("</parameter>", "").strip()
            args_dict[p_name.strip()] = int(clean_val) if clean_val.isdigit() else clean_val
                
        if not args_dict and "{" in full_response_text:
            try:
                json_match = re.search(r'(\{.*?\})', full_response_text, re.DOTALL)
                if json_match:
                    args_dict = json.loads(json_match.group(1))
            except Exception:
                pass

        final_json_args = json.dumps(args_dict, ensure_ascii=False)
        logger.info(f"Generated Tool Call: '{tool_name}' with args: {final_json_args}")
        
        # Именованная передача вызова функции
        asyncio.run_coroutine_threadsafe(
            queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, tool_name=tool_name, tool_args=final_json_args, finish_reason="tool_calls")), 
            loop
        )
    else:
        # Именованная передача обычного завершения текста
        asyncio.run_coroutine_threadsafe(
            queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, finish_reason="stop")), 
            loop
        )
        
    end_time = time.time()
    mem_after = mx.get_active_memory()
    cache_nbytes = max(0, mem_after - mem_before)
    generation_time = end_time - generation_start_time
    tokens_per_sec = tokens_count / generation_time if generation_time > 0 else 0
    
    logger.info(f"Prompt Cache VRAM Footprint: {cache_nbytes / 1e9:.4f} GB")
    logger.info(f"Generated {tokens_count} tokens in {generation_time:.2f} s ({tokens_per_sec:.2f} text tokens/s)")
    
    # Сигнализируем мосту о завершении генерации
    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

async def async_queue_bridge(model, tokenizer, prompt_ids, max_tokens, request_id, has_tools, prefill_step_size, global_cache, model_name):
    """Асинхронный мост, связывающий фоновый поток ОС и асинхронный стрим FastAPI."""
    from starlette.concurrency import run_in_threadpool
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    prompt_tokens_len = len(prompt_ids)
    
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
        queue.task_done()
    yield "data: [DONE]\n\n"
