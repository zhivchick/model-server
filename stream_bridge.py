import sys
import asyncio
import re
import time
import logging
import json
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

            generation_start_time = time.time()
            logger.debug("Entering main Decoding token stream loop...")

            for response in generator_instance:
                token = response.text
                full_response_text += token
                tokens_count += 1
                
                in_tool_call = parser.parse_chunk(token) if has_tools else False
                if in_tool_call:
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    continue

                sys.stdout.write(token)
                sys.stdout.flush()
                
                asyncio.run_coroutine_threadsafe(
                    queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, content=token)), loop
                )

        print()
        generation_time = time.time() - generation_start_time
        current_decode_speed = tokens_count / generation_time if generation_time > 0 else 0.0

        # Main stream parsing is done. Now evaluate the firewall state.
        is_raw_json_tool = False
        json_tool_name = None
        json_tool_args = "{}"
        
        if has_tools and not parser.in_tool_call:
            start_idx = full_response_text.find("{")
            if start_idx != -1 and "arguments" in full_response_text:
                potential_json = full_response_text[start_idx:].strip()
                if potential_json.endswith("```"):
                    potential_json = potential_json[:-3].strip()
                try:
                    parsed_json = json.loads(potential_json)
                    if "name" in parsed_json and "arguments" in parsed_json:
                        is_raw_json_tool = True
                        json_tool_name = parsed_json["name"]
                        args_obj = parsed_json["arguments"]
                        json_tool_args = json.dumps(args_obj, ensure_ascii=False) if isinstance(args_obj, dict) else str(args_obj)
                except Exception: pass

        if has_tools and (parser.in_tool_call or is_raw_json_tool):
            if is_raw_json_tool:
                extracted_args = json.loads(json_tool_args)
                tool_invocation_name = json_tool_name
            else:
                extracted_args = _parse_xml_arguments(full_response_text)
                tool_invocation_name = parser.tool_name
                
            # 🔍 Передаем данные в файрвол для анализа петель
            t_name, t_args = anti_loop_engine.evaluate_and_process(full_response_text, tool_invocation_name, extracted_args)
            
            # 🎯 ЕСЛИ ПЕТЛИ НЕТ (t_name совпадает с оригиналом) — МЫ НЕ ДOКИДЫВАЕМ TOOL_CALLS ВДOГOНКУ!
            # Мы просто закрываем текстовый стрим флагом "stop", позволяя Goose нативно сожрать чистый XML!
            if t_name == tool_invocation_name:
                asyncio.run_coroutine_threadsafe(
                    queue.put(build_streaming_chunk(
                        request_id=request_id, model_name=model_name, finish_reason="stop", 
                        prompt_len=prompt_tokens_len, completion_len=tokens_count
                    )), loop
                )
            else:
                # 🚨 ПЕТЛЯ ОБНАРУЖЕНА! Вот тут мы жестко перебиваем стрим и вбрасываем ошибку shell exit 1
                asyncio.run_coroutine_threadsafe(
                    queue.put(build_streaming_chunk(
                        request_id=request_id, model_name=model_name, tool_name=t_name, tool_args=t_args, 
                        finish_reason="tool_calls", prompt_len=prompt_tokens_len, completion_len=tokens_count
                    )), loop
                )
        else:
            # Обычный текстовый ответ без инструментов
            asyncio.run_coroutine_threadsafe(
                queue.put(build_streaming_chunk(
                    request_id=request_id, model_name=model_name, finish_reason="stop", 
                    prompt_len=prompt_tokens_len, completion_len=tokens_count
                )), loop
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
