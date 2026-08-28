import uuid
import argparse
import logging
import datetime
import re
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

import mlx_lm
from goose_hooks import apply_pre_call_hooks
from stream_bridge import async_queue_bridge

# Настройка парсера аргументов командной строки
parser = argparse.ArgumentParser(description="Stateful Multi-Session MLX Server")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--max-tokens", type=int, default=4096)
parser.add_argument("--prefill-step-size", type=int, default=512)
parser.add_argument("--log-level", type=str, default="info", choices=["info", "debug", "warning", "error"])
args, unknown = parser.parse_known_args()

# Конфигурация логгера
numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("mlx_lm_server")

# ANSI-палитра для красивой раскраски консоли
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RESET = "\033[0m"

print(f"📦 Loading model: {args.model}...")
model, tokenizer = mlx_lm.load(args.model)

def make_persistent_cache():
    from mlx_lm.models.cache import make_prompt_cache
    try:
        cache = make_prompt_cache(model)
        for layer_cache in cache:
            if hasattr(layer_cache, "bits"): layer_cache.bits = 3
            if hasattr(layer_cache, "quantized_start"): layer_cache.quantized_start = 64
        return cache
    except Exception:
        from mlx_lm.models.cache import KVCache
        return [KVCache() for _ in range(len(model.layers) if hasattr(model, "layers") else 32)]

# Инициализируем аппаратный кэш Агента
AGENT_CACHE = make_persistent_cache()
PREVIOUS_AGENT_IDS = []

print("\n" + "="*60 + f"\n🚀 [Goose Server] Time-Locked Production Engine Active!\n💡 Mode: Agent Persistent / Utility Ephemeral\nLog Level: {args.log_level.upper()}\n" + "="*60 + "\n")
app = FastAPI()

def find_longest_common_token_prefix(list1: list, list2: list) -> int:
    min_len = min(len(list1), len(list2))
    for i in range(min_len):
        if list1[i] != list2[i]: return i
    return min_len

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global AGENT_CACHE, PREVIOUS_AGENT_IDS
    
    body = await request.json()
    max_tokens = body.get("max_tokens", args.max_tokens)
    request_id = f"chatcmpl-{uuid.uuid4()}"
    
    fixed_messages, template_kwargs = apply_pre_call_hooks(body)
    has_tools = body.get("tools") is not None
    
    # Железная сегрегация ролей по наличию инструментов
    is_agent = has_tools

    full_prompt_string = tokenizer.apply_chat_template(fixed_messages, **template_kwargs)
    
    # 🎯 «ЗАМOК ВРЕМЕНИ»: Маскируем ежеминутно тикающий тег Goose.
    # Заменяем цифры времени на константу строго для токенизации и сравнения префиксов!
    prompt_for_cache_comparison = re.sub(
        r"<current-time>.*?</current-time>", 
        "<current-time>STATIC_TIME_LOCK</current-time>", 
        full_prompt_string
    )
    
    current_prompt_ids = tokenizer.encode(prompt_for_cache_comparison)
    total_prompt_len = len(current_prompt_ids)

    # СЦЕНАРИЙ 1: СЛУЖЕБНЫЙ ЗАПРОС (UTILITY) -> ИЗОЛИРОВАННЫЙ КЭШ
    if not is_agent:
        logger.info(f"POST /v1/chat/completions | Target: [UTILITY] (ID: {request_id})")
        logger.info(f"🧹 [Utility Ephemeral] Context maps evaluated: {total_prompt_len} tokens.")
        ephemeral_cache = make_persistent_cache()
        
        return StreamingResponse(
            async_queue_bridge(
                model, tokenizer, current_prompt_ids, max_tokens, request_id, has_tools, 
                args.prefill_step_size, ephemeral_cache, args.model, total_prompt_len, None
            ),
            media_type="text/event-stream"
        )

    # СЦЕНАРИЙ 2: БОЕВОЙ АГЕНТ (AGENT) -> РАБОТАЕТ ВЕЧНЫЙ КЭШ
    logger.info(f"POST /v1/chat/completions | Target: [AGENT] (ID: {request_id})")
    
    if PREVIOUS_AGENT_IDS:
        matched_tokens_len = find_longest_common_token_prefix(PREVIOUS_AGENT_IDS, current_prompt_ids)
        
        # Легкая трассировка номеров токенов уходит в DEBUG
        if matched_tokens_len > 0 and matched_tokens_len < len(PREVIOUS_AGENT_IDS) and matched_tokens_len < len(current_prompt_ids):
            token_prev = PREVIOUS_AGENT_IDS[matched_tokens_len]
            token_curr = current_prompt_ids[matched_tokens_len]
            logger.debug(f"🔍 [CACHE TRACE] Split index: {matched_tokens_len} | Prev Token ID: {token_prev} vs Curr Token ID: {token_curr}")

        # 🎯 SMART DIFF TRIGGER: Если кэш Агента обвалился более чем на 5000 токенов
        cache_drop_size = len(PREVIOUS_AGENT_IDS) - matched_tokens_len
        if cache_drop_size >= 5000:
            timestamp = datetime.datetime.now().strftime("%H_%M_%S")
            file_saved = f"saved_drop_{timestamp}.txt"
            file_new = f"new_drop_{timestamp}.txt"
            try:
                with open(file_saved, "w", encoding="utf-8") as f:
                    f.write(tokenizer.decode(PREVIOUS_AGENT_IDS))
                with open(file_new, "w", encoding="utf-8") as f:
                    f.write(tokenizer.decode(current_prompt_ids))
                logger.warning(
                    f"{C_YELLOW}⚠️ [CRITICAL CACHE DROP] Context collapsed by {cache_drop_size} tokens! "
                    f"Dumped snapshots: diff {file_saved} {file_new}{C_RESET}"
                )
            except Exception as e:
                logger.debug(f"Failed to write smart diff files: {str(e)}")

        if matched_tokens_len > 300:
            prompt_ids_chunk = current_prompt_ids[matched_tokens_len:]
            
            if len(prompt_ids_chunk) == 0:
                matched_tokens_len -= 1
                prompt_ids_chunk = [current_prompt_ids[-1]]
            
            # Сдвигаем offset Metal API только если произошел честный глубокий откат истории
            if matched_tokens_len < len(PREVIOUS_AGENT_IDS):
                for layer_cache in AGENT_CACHE:
                    if hasattr(layer_cache, "offset"): layer_cache.offset = matched_tokens_len
                    elif hasattr(layer_cache, "step"): layer_cache.step = matched_tokens_len
            
            # Подсвечиваем сочным зеленым цветом цифру совпавшего контекста
            logger.info(f"🎯 [Cache AGENT Hit] Reused context: {C_GREEN}{matched_tokens_len}{C_RESET} tokens. Evaluating delta remainder: {len(prompt_ids_chunk)} tokens.")
            
            PREVIOUS_AGENT_IDS = current_prompt_ids
            
            return StreamingResponse(
                async_queue_bridge(
                    model, tokenizer, prompt_ids_chunk, max_tokens, request_id, has_tools, 
                    args.prefill_step_size, AGENT_CACHE, args.model, total_prompt_len, None
                ),
                media_type="text/event-stream"
            )

    logger.info(f"🧹 [Cache AGENT Miss] Full evaluation required: {total_prompt_len} tokens.")
    AGENT_CACHE = make_persistent_cache()
    PREVIOUS_AGENT_IDS = current_prompt_ids
    
    return StreamingResponse(
        async_queue_bridge(
            model, tokenizer, current_prompt_ids, max_tokens, request_id, has_tools, 
            args.prefill_step_size, AGENT_CACHE, args.model, total_prompt_len, None
        ),
        media_type="text/event-stream"
    )

# Запуск Uvicorn
uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
