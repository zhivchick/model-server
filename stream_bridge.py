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

def _log_stream_start(request_id: str, is_utility: bool, prompt_len: int):
    """Logs the initialization and lock acquisition for the GPU thread."""
    print("-" * 60)
    if is_utility:
        logger.info(f"[GPU LOCK] Utility request {request_id} acquired exclusive GPU access.")
        print(f"⚙️ [Utility Stream | Context: {prompt_len} tokens] ", end="")
    else:
        logger.info(f"[GPU LOCK] Main agent {request_id} acquired exclusive GPU access.")
        print(f"🖥️ [Agent Stream | Context: {prompt_len} tokens] ", end="")
    sys.stdout.flush()

def _process_tool_call(full_text: str, parser: QwenXmlParser) -> tuple:
    """Extracts tool name and arguments from the full generated response text."""
    # 🎯 Extract function name from full response text
    func_match = re.search(r'<function=([^>]+)>', full_text)
    if func_match:
        tool_name = func_match.group(1).strip()
    else:
        tool_name = parser.tool_name if parser.tool_name else "shell"

    # 🎯 FIX: Restore the original, working regex parser directly over full_text
    param_matches = re.findall(r'<parameter=([^>]+)>(.*?)(?:</parameter>|$)', full_text, re.DOTALL)
    
    args_dict = {}
    for p_name, p_val in param_matches:
        clean_val = p_val.replace("</parameter>", "").strip()
        if clean_val.isdigit():
            args_dict[p_name.strip()] = int(clean_val)
        else:
            args_dict[p_name.strip()] = clean_val

    # Fallback to extract raw JSON if XML structure is missing but JSON block exists
    if not args_dict and "{" in full_text:
        try:
            json_match = re.search(r'(\{.*?\})', full_text, re.DOTALL)
            if json_match:
                args_dict = json.loads(json_match.group(1))
        except Exception:
            pass

    final_json_args = json.dumps(args_dict, ensure_ascii=False)
    logger.info(f"Generated Tool Call: '{tool_name}' with args: {final_json_args}")
    return tool_name, final_json_args

def sync_generation_worker(model, tokenizer, prompt_ids, max_tokens, request_id, has_tools, prefill_step_size, global_cache, queue, loop, model_name, prompt_tokens_len):
    """Background worker with absolute GPU thread isolation and token streaming."""
    global GPU_THREAD_LOCK
    GPU_THREAD_LOCK.acquire()
    
    is_utility = not has_tools
    _log_stream_start(request_id, is_utility, prompt_tokens_len)
    
    try:
        full_response_text = ""
        tokens_count = 0
        parser = QwenXmlParser()
        
        # Track precise prefill evaluation time
        prefill_start_time = time.perf_counter()
        
        generator_instance = stream_generate(
            model, tokenizer, 
            prompt=prompt_ids, 
            max_tokens=max_tokens, 
            prompt_cache=global_cache, 
            prefill_step_size=prefill_step_size
        )

        # ⚡️ Evaluate and process the first token (Prefill phase)
        try:
            first_response = next(generator_instance)
            prefill_time = time.perf_counter() - prefill_start_time
            
            token = first_response.text
            full_response_text += token
            tokens_count += 1
            
            # Print evaluation metrics for the evaluated token chunk
            chunk_len = len(prompt_ids)
            prefill_speed = chunk_len / prefill_time if prefill_time > 0 else 0
            logger.info(f"Prefill finished: evaluated {chunk_len} tokens in {prefill_time:.4f}s ({prefill_speed:.2f} tok/s)")
            
            is_tool = parser.parse_chunk(token) if has_tools else False
            if not is_tool:
                asyncio.run_coroutine_threadsafe(
                    queue.put(build_streaming_chunk(request_id=request_id, model_name=model_name, content=token)), 
                    loop
                )
        except StopIteration:
            return

        generation_start_time = time.time()

        # 🏃‍♂️ Token streaming loop (Decoding phase)
        for response in generator_instance:
            token = response.text
            full_response_text += token
            tokens_count += 1
            
            in_tool_call = parser.parse_chunk(token) if has_tools else False
            
            # Intercept XML tool calls to prevent raw tags from breaking the Goose UI markdown
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

        print() # Close the stream console print line
        
        # 🎯 Total context length mapping directly from server state
        total_prompt_context_len = prompt_tokens_len

        # Emit the final completion chunk with accurate Goose Usage telemetry
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
        if generation_time > 0:
            decoding_speed = tokens_count / generation_time
            logger.info(f"[TRACKER METRICS] Reported context size to Goose: {total_prompt_context_len} tokens.")
            logger.info(f"Decoding finished: generated {tokens_count} tokens in {generation_time:.2f}s ({decoding_speed:.2f} tok/s)")

    except Exception as e:
        logger.exception(f"Critical exception inside GPU worker execution loop: {str(e)}")
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        GPU_THREAD_LOCK.release()
        logger.info(f"[GPU RELEASED] Request {request_id} execution finalized.")
        print("-" * 60)

async def async_queue_bridge(model, tokenizer, prompt_ids, max_tokens, request_id, has_tools, prefill_step_size, global_cache, model_name, prompt_tokens_len):
    """Asynchronous transit gateway bridging threadpool queue execution to FastAPI SSE stream."""
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
