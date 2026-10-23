import sys
import asyncio
import time
import logging
import mlx.core as mx
from mlx_lm.generate import stream_generate
from response_formatters import build_streaming_chunk
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


def sync_generation_worker(
    model,
    tokenizer,
    prompt_ids,
    max_tokens,
    request_id,
    has_tools,
    prefill_step_size,
    global_cache,
    queue,
    loop,
    model_name,
    prompt_tokens_len,
    repetition_penalty=1.0,
    kv_bits=8,
    kv_group_size=64,
    quantized_kv_start=32768,
    on_generation_complete=None
):
    """Background worker executing inside Starlette run_in_threadpool context."""
    logger.debug(f"🚀 Entering sync_generation_worker for request: {request_id}")
    
    mx.set_default_device(mx.gpu)
    is_utility = not has_tools
    
    # Стартовый лог и замер времени префилла
    if not is_utility and len(prompt_ids) <= prefill_step_size:
        _log_stream_start(request_id, is_utility, prompt_tokens_len)
    
    current_prefill_speed = 0.0
    current_decode_speed = 0.0
    chunk_len = len(prompt_ids)
    generated_token_ids = []
    
    try:
        full_response_text = ""
        tokens_count = 0
        parser = QwenXmlParser()
        
        prefill_start_time = time.perf_counter()
        
        with mx.StreamContext(mx.default_stream(mx.gpu)):
            
            # ⏳ Коллбэк для отображения живого прогресса префилла в консоли сервера
            def _prefill_progress(processed: int, total: int):
                if total > prefill_step_size and not is_utility:
                    pct = int(processed * 100 / total)
                    elapsed = time.perf_counter() - prefill_start_time
                    spd = processed / elapsed if elapsed > 0 else 0.0
                    sys.stdout.write(f"\r⏳ Prefill: [{pct:3d}%] {processed}/{total} tokens ({spd:.0f} t/s)... ")
                    sys.stdout.flush()

            gen_kwargs = {
                "prompt": prompt_ids,
                "max_tokens": max_tokens,
                "prompt_cache": global_cache,
                "prefill_step_size": prefill_step_size,
                "prompt_progress_callback": _prefill_progress,
            }
            
            # Нативное квантование KV-кэша в mlx_lm
            if kv_bits and kv_bits > 0:
                gen_kwargs["kv_bits"] = kv_bits
                gen_kwargs["kv_group_size"] = kv_group_size
                gen_kwargs["quantized_kv_start"] = quantized_kv_start

            if repetition_penalty and repetition_penalty > 1.0:
                try:
                    from mlx_lm.sample_utils import make_logits_processors
                    processors = make_logits_processors(
                        repetition_penalty=repetition_penalty,
                        repetition_context_size=64
                    )
                    if processors:
                        gen_kwargs["logits_processors"] = processors
                except Exception:
                    pass

            generator_instance = stream_generate(model, tokenizer, **gen_kwargs)

            try:
                first_response = next(generator_instance)
                
                # Если префилл был большим, стираем строку прогресса и выводим заголовок агента
                if chunk_len > prefill_step_size and not is_utility:
                    sys.stdout.write("\r" + " " * 75 + "\r")
                    _log_stream_start(request_id, is_utility, prompt_tokens_len)
                elif is_utility:
                    _log_stream_start(request_id, is_utility, prompt_tokens_len)

                prefill_time = time.perf_counter() - prefill_start_time
                token = first_response.text
                if hasattr(first_response, "token"):
                    generated_token_ids.append(int(first_response.token))
                full_response_text += token
                tokens_count += 1
                
                current_prefill_speed = chunk_len / prefill_time if prefill_time > 0 else 0.0
                
                is_tool = parser.parse_chunk(token) if has_tools else False
                sys.stdout.write(token)
                sys.stdout.flush()
                if not is_tool:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, content=token)),
                        loop
                    )
            except StopIteration:
                return

            generation_start_time = time.time()

            for response in generator_instance:
                token = response.text
                if hasattr(response, "token"):
                    generated_token_ids.append(int(response.token))
                full_response_text += token
                tokens_count += 1
                
                in_tool_call = parser.parse_chunk(token) if has_tools else False
                sys.stdout.write(token)
                sys.stdout.flush()

                if in_tool_call:
                    continue

                asyncio.run_coroutine_threadsafe(
                    queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, content=token)),
                    loop
                )

        print()
        generation_time = time.time() - generation_start_time
        current_decode_speed = tokens_count / generation_time if generation_time > 0 else 0.0

        if on_generation_complete and generated_token_ids:
            on_generation_complete(generated_token_ids)

        # Маршрутизация и вызов радара анти-петли
        if has_tools and parser.in_tool_call:
            extracted_args = parser.extract_arguments_dict()
            tool_invocation_name = parser.tool_name

            t_name, t_args = anti_loop_engine.evaluate_and_process(
                full_response_text, tool_invocation_name, extracted_args
            )
            
            if t_name is None:
                asyncio.run_coroutine_threadsafe(
                    queue.put(build_streaming_chunk(
                        request_id=request_id,
                        model_name=model_name,
                        content=f"\n\n🛑 {t_args}\n",
                        finish_reason="stop",
                        prompt_len=prompt_tokens_len,
                        completion_len=tokens_count
                    )),
                    loop
                )
            else:
                asyncio.run_coroutine_threadsafe(
                    queue.put(build_streaming_chunk(
                        request_id=request_id,
                        model_name=model_name,
                        tool_name=t_name,
                        tool_args=t_args,
                        finish_reason="tool_calls",
                        prompt_len=prompt_tokens_len,
                        completion_len=tokens_count
                    )),
                    loop
                )
        else:
            asyncio.run_coroutine_threadsafe(
                queue.put(build_streaming_chunk(
                    request_id=request_id,
                    model_name=model_name,
                    finish_reason="stop",
                    prompt_len=prompt_tokens_len,
                    completion_len=tokens_count
                )),
                loop
            )

        perf_tracker.record_metrics(
            current_prefill_speed,
            current_decode_speed,
            total_context_len=prompt_tokens_len,
            prompt_chunk_len=chunk_len,
            completion_len=tokens_count
        )

    except Exception as e:
        logger.exception(f"Critical exception inside GPU worker execution loop: {str(e)}")
    finally:
        # Очищаем неиспользуемые буферы пула Metal после каждого запроса
        try:
            mx.clear_cache()
        except Exception:
            pass
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        logger.info(f"[GPU RELEASED] Request {request_id} execution finalized.")
        print("-" * 60)


async def async_queue_bridge(
    model,
    tokenizer,
    prompt_ids,
    max_tokens,
    request_id,
    has_tools,
    prefill_step_size,
    global_cache,
    model_name,
    prompt_tokens_len,
    repetition_penalty=1.0,
    kv_bits=8,
    kv_group_size=64,
    quantized_kv_start=32768,
    on_generation_complete=None
):
    from starlette.concurrency import run_in_threadpool
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    asyncio.create_task(
        run_in_threadpool(
            sync_generation_worker,
            model,
            tokenizer,
            prompt_ids,
            max_tokens,
            request_id,
            has_tools,
            prefill_step_size,
            global_cache,
            queue,
            loop,
            model_name,
            prompt_tokens_len,
            repetition_penalty,
            kv_bits,
            kv_group_size,
            quantized_kv_start,
            on_generation_complete
        )
    )
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk
    yield "data: [DONE]\n\n"
