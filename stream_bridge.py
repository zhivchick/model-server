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

# Import sub-modules logic blocks
from perf_tracker import perf_tracker
from anti_loop import anti_loop_engine

logger = logging.getLogger("mlx_lm_server.bridge")
GPU_THREAD_LOCK = threading.Lock()

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
    """Helper method to parse raw text attributes into clean dictionary layouts."""
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

def sync_generation_worker(model, tokenizer, prompt_ids, max_tokens, request_id, has_tools, prefill_step_size, global_cache, queue, loop, model_name, prompt_tokens_len):
    """Background worker with absolute GPU thread isolation and token streaming."""
    global GPU_THREAD_LOCK
    GPU_THREAD_LOCK.acquire()
    
    mx.set_default_device(mx.gpu)
    is_utility = not has_tools
    _log_stream_start(request_id, is_utility, prompt_tokens_len)
    
    current_prefill_speed = 0.0
    current_decode_speed = 0.0
    
    try:
        full_response_text = ""
        tokens_count = 0
        parser = QwenXmlParser()
        
        # Safely evaluate persistent memory state maps inside this specific thread context
        try:
            if isinstance(global_cache, list) and len(global_cache) > 0:
                if getattr(global_cache, "keys", None) is not None:
                    mx.eval([c.state for c in global_cache])
        except Exception: pass
        
        prefill_start_time = time.perf_counter()
        
        # Run inside hard localized stream manager context boundary
        with mx.StreamContext(mx.default_stream(mx.gpu)):
            generator_instance = stream_generate(
                model, tokenizer, prompt=prompt_ids, max_tokens=max_tokens, 
                prompt_cache=global_cache, prefill_step_size=prefill_step_size
            )

            try:
                first_response = next(generator_instance)
                prefill_time = time.perf_counter() - prefill_start_time
                token = first_response.text
                full_response_text += token
                tokens_count += 1
                
                chunk_len = len(prompt_ids)
                current_prefill_speed = chunk_len / prefill_time if prefill_time > 0 else 0.0
                
                is_tool = parser.parse_chunk(token) if has_tools else False
                if not is_tool:
                    asyncio.run_coroutine_threadsafe(
                        queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, content=token)), loop
                    )
            except StopIteration: return

            generation_start_time = time.time()

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

        if has_tools and parser.in_tool_call:
            extracted_args = _parse_xml_arguments(full_response_text)
            
            # 🎯 Route directly to the isolated Anti-Loop Sub-Module Engine
            t_name, t_args = anti_loop_engine.evaluate_and_process(full_response_text, parser.tool_name, extracted_args)
            
            asyncio.run_coroutine_threadsafe(
                queue.put(build_streaming_chunk(
                    request_id=request_id, model_name=model_name, tool_name=t_name, tool_args=t_args, 
                    finish_reason="tool_calls", prompt_len=prompt_tokens_len, completion_len=tokens_count
                )), loop
            )
        else:
            asyncio.run_coroutine_threadsafe(
                queue.put(build_streaming_chunk(
                    request_id=request_id, model_name=model_name, finish_reason="stop", 
                    prompt_len=prompt_tokens_len, completion_len=tokens_count
                )), loop
            )
            
        # 🎯 Route filtered telemetry straight to the isolated Performance Tracker Sub-Module
        perf_tracker.record_metrics(
            current_prefill_speed, 
            current_decode_speed, 
            total_context_len=prompt_tokens_len,
            prompt_chunk_len=chunk_len,       # Real size of evaluated tokens chunk
            completion_len=tokens_count        # Real size of generated response
        )

    except Exception as e:
        logger.exception(f"Critical exception inside GPU worker execution loop: {str(e)}")
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        GPU_THREAD_LOCK.release()
        logger.info(f"[GPU RELEASED] Request {request_id} execution finalized.")
        print("-" * 60)

async def async_queue_bridge(model, tokenizer, prompt_ids, max_tokens, request_id, has_tools, prefill_step_size, global_cache, model_name, prompt_tokens_len):
    from starlette.concurrency import run_in_threadpool
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    asyncio.create_task(
        run_in_threadpool(
            sync_generation_worker, model, tokenizer, prompt_ids, max_tokens, request_id, has_tools, 
            prefill_step_size, global_cache, queue, loop, model_name, prompt_tokens_len
        )
    )
    while True:
        chunk = await queue.get()
        if chunk is None: break
        yield chunk
    yield "data: [DONE]\n\n"
