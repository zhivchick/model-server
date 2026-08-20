import uuid
import argparse
import logging
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

import mlx_lm
from goose_hooks import apply_pre_call_hooks
from stream_bridge import async_queue_bridge

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("mlx_lm_server")

parser = argparse.ArgumentParser(description="Stateful Multi-Session MLX Server")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--max-tokens", type=int, default=4096)
parser.add_argument("--prefill-step-size", type=int, default=512)
args, unknown = parser.parse_known_args()

print(f"📦 Loading model: {args.model}...")
model, tokenizer = mlx_lm.load(args.model)

def make_persistent_cache():
    if hasattr(model, "make_cache"):
        cache = model.make_cache()
        for layer_cache in cache:
            if hasattr(layer_cache, "bits"): layer_cache.bits = 3
            if hasattr(layer_cache, "quantized_start"): layer_cache.quantized_start = 64
        return cache
    from mlx_lm.models.base import create_kv_cache
    return create_kv_cache(model)

# 🎯 Isolated VRAM memory spaces (Agent logic loop vs background processing utility tasks)
GLOBAL_CACHE_REGISTRY = {
    "agent": make_persistent_cache(),
    "utility": make_persistent_cache()
}

# 🎯 Persistent token ID sequence storage ensuring strict tokenized context alignment
PREVIOUS_IDS_REGISTRY = {
    "agent": [],
    "utility": []
}

print("\n" + "="*60 + "\n🚀 [Goose Stateful Server] Successfully Initialization Complete!\n💡 Framework: Isolated Multi-Session VRAM Management (Apple Style)\n" + "="*60 + "\n")
app = FastAPI()

def find_longest_common_token_prefix(list1: list, list2: list) -> int:
    """Calculates matching prefix depth natively operating on Integer Token IDs sequences."""
    min_len = min(len(list1), len(list2))
    for i in range(min_len):
        if list1[i] != list2[i]:
            return i
    return min_len

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global GLOBAL_CACHE_REGISTRY, PREVIOUS_IDS_REGISTRY
    body = await request.json()
    max_tokens = body.get("max_tokens", args.max_tokens)
    request_id = f"chatcmpl-{uuid.uuid4()}"
    
    # Process normalization and system-wide reasoning suppression hooks
    fixed_messages, template_kwargs = apply_pre_call_hooks(body)
    has_tools = body.get("tools") is not None
    session_key = "agent" if has_tools else "utility"
    
    # Accurate single-pass full pipeline prompt tokenization
    full_prompt_string = tokenizer.apply_chat_template(fixed_messages, **template_kwargs)
    current_prompt_ids = tokenizer.encode(full_prompt_string)
    total_prompt_len = len(current_prompt_ids)

    logger.info(f"POST /v1/chat/completions | Session target: [{session_key.upper()}] (ID: {request_id})")

    prev_ids = PREVIOUS_IDS_REGISTRY[session_key]
    active_cache = GLOBAL_CACHE_REGISTRY[session_key]
    
    # Match against active session history cache using token-level tracking
    if prev_ids:
        matched_tokens_len = find_longest_common_token_prefix(prev_ids, current_prompt_ids)
        
        # Prefix retention threshold constraint (prevents offset thrashing for minor token deltas)
        if matched_tokens_len > 300:
            prompt_ids_chunk = current_prompt_ids[matched_tokens_len:]
            
            # 🎯 FIX FOR 100% CACHE HIT (0 tokens delta error):
            # If there are no new tokens, steal the last token from the cache to trigger MLX generator properly
            if len(prompt_ids_chunk) == 0:
                matched_tokens_len -= 1
                prompt_ids_chunk = [current_prompt_ids[-1]]
                logger.info("⚠️ 100% Cache Hit detected (0 new tokens). Adjusting cache offset by -1 to trigger generation.")
            
            # Sync C++ level hardware execution offsets across Apple Metal context cache layers
            for layer_cache in active_cache:
                if hasattr(layer_cache, "offset"):
                    layer_cache.offset = matched_tokens_len
                elif hasattr(layer_cache, "step"):
                    layer_cache.step = matched_tokens_len
            
            logger.info(f"🎯 [Cache {session_key.upper()} Hit] Reused context: {matched_tokens_len} tokens. Evaluating delta remainder: {len(prompt_ids_chunk)} tokens.")
            PREVIOUS_IDS_REGISTRY[session_key] = current_prompt_ids

            
            return StreamingResponse(
                async_queue_bridge(
                    model, tokenizer, prompt_ids_chunk, max_tokens, request_id, has_tools, 
                    args.prefill_step_size, active_cache, args.model, 
                    prompt_tokens_len=total_prompt_len  # 🎯 Context sum metrics: matched + evaluated delta
                ),
                media_type="text/event-stream"
            )

    # Cache Miss handling routine
    if len(fixed_messages) <= 2:
        logger.info(f"🧹 [Cache {session_key.upper()} Reset] Fresh execution stream chain initialization. Buffers cleared.")
    else:
        logger.info(f"🧹 [Cache {session_key.upper()} Miss] Session history prefix mismatch tracking. Full contextual evaluation required: {total_prompt_len} tokens.")
        
    active_cache = make_persistent_cache()
    GLOBAL_CACHE_REGISTRY[session_key] = active_cache
    PREVIOUS_IDS_REGISTRY[session_key] = current_prompt_ids
    
    return StreamingResponse(
        async_queue_bridge(
            model, tokenizer, current_prompt_ids, max_tokens, request_id, has_tools, 
            args.prefill_step_size, active_cache, args.model, 
            prompt_tokens_len=total_prompt_len  # 🎯 Context metrics allocation
        ),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
