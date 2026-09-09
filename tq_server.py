import uuid
import argparse
import logging
import datetime
import re
import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

import mlx_lm
from goose_hooks import apply_pre_call_hooks
from stream_bridge import async_queue_bridge

# ------------------------------------------------------------
# ⚙️ НАСТРОЙКИ И ИНИЦИАЛИЗАЦИЯ
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Stateful Multi-Session MLX Server")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--max-tokens", type=int, default=4096)
parser.add_argument("--prefill-step-size", type=int, default=512)
parser.add_argument("--log-level", type=str, default="info", choices=["info", "debug", "warning", "error"])
args, unknown = parser.parse_known_args()

numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("mlx_lm_server")

C_GREEN, C_YELLOW, C_RESET = "\033[92m", "\033[93m", "\033[0m"

print(f"📦 Loading model: {args.model}...")
model, tokenizer = mlx_lm.load(args.model)

# Реестры аппаратно разделенных кэшей Apple Metal
AGENT_CACHE = [mlx_lm.models.cache.KVCache() for _ in range(32)] # Ленивая инициализация в функции
PREVIOUS_AGENT_IDS = []
COMPACTION_CACHE = [mlx_lm.models.cache.KVCache() for _ in range(32)]
PREVIOUS_COMPACTION_IDS = []
ASYNC_SERVER_LOCK = asyncio.Lock()

app = FastAPI()

# ------------------------------------------------------------
# 🌐 ОСНОВНОЙ СЕТЕВОЙ ЭНДПОИНТ (МАКСИМАЛЬНО КОРОТКИЙ)
# ------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global AGENT_CACHE, PREVIOUS_AGENT_IDS, COMPACTION_CACHE, PREVIOUS_COMPACTION_IDS
    
    await ASYNC_SERVER_LOCK.acquire()
    try:
        body = await request.json()
        request_id = f"chatcmpl-{uuid.uuid4()}"
        
        fixed_messages, template_kwargs = apply_pre_call_hooks(body)
        is_agent = body.get("tools") is not None

        # 🧠 Вызываем процедуры очистки и токенизации
        fixed_messages = _apply_hardware_time_lock(fixed_messages)
        current_prompt_ids, total_prompt_len = _render_and_tokenize(fixed_messages, template_kwargs)
        _print_pipeline_telemetry(request_id, is_agent, total_prompt_len)

        # Маршрутизация сценариев выполнения
        if not is_agent:
            return _handle_utility_scenarios(current_prompt_ids, total_prompt_len, request_id, body.get("max_tokens", args.max_tokens))
        else:
            return _handle_agent_scenarios(current_prompt_ids, total_prompt_len, request_id, body.get("max_tokens", args.max_tokens))

    except Exception as e:
        if ASYNC_SERVER_LOCK.locked(): ASYNC_SERVER_LOCK.release()
        raise e

# ------------------------------------------------------------
# 🧠 ИЗОЛИРОВАННЫЕ ПРОЦЕДУРЫ ОБРАБОТКИ КОНТЕКСТА
# ------------------------------------------------------------
def _apply_hardware_time_lock(messages: list) -> list:
    """Замораживает динамические теги времени в JSON до вызова токенизатора."""
    for msg in messages:
        if "content" in msg and isinstance(msg["content"], str):
            content_str = msg["content"]
            content_str = re.sub(r"<current-time>.*?</current-time>", "<current-time>STATIC_TIME_LOCK</current-time>", content_str)
            content_str = re.sub(r"<compaction>.*?</compaction>", "<compaction>STATIC_COMPACTION_LOCK</compaction>", content_str)
            msg["content"] = content_str
    return messages

def _render_and_tokenize(messages: list, template_kwargs: dict) -> tuple:
    """Применяет Jinja-шаблон строго 1 раз и кодирует без скрытых токенов BOS."""
    full_prompt_string = tokenizer.apply_chat_template(messages, **template_kwargs)
    current_prompt_ids = tokenizer.encode(full_prompt_string, add_special_tokens=False)
    return current_prompt_ids, len(current_prompt_ids)

def _print_pipeline_telemetry(request_id: str, is_agent: bool, total_prompt_len: int):
    """Выводит плоскую карту координат промпта в консоль."""
    print(f"\n{'='*20} PROMPT DATA {'='*20}")
    print(f"Request: {request_id} | Mode: {'[AGENT]' if is_agent else '[UTILITY]'} | Tokens: {total_prompt_len}")
    print(f"{'='*53}\n")
    import sys; sys.stdout.flush()

# ------------------------------------------------------------
# ⚡ ИЗОЛИРОВАННЫЕ СЦЕНАРИИ КЭШИРОВАНИЯ И ИНФЕРЕНСА
# ------------------------------------------------------------
def _handle_utility_scenarios(current_prompt_ids: list, total_prompt_len: int, request_id: str, max_tokens: int):
    """Управляет фоновыми запросами и автотайтлами сессий."""
    global COMPACTION_CACHE, PREVIOUS_COMPACTION_IDS
    logger.info(f"POST /v1/chat/completions | Target: [UTILITY] (ID: {request_id})")
    
    # Режим компакта истории (>3000 токенов)
    if total_prompt_len > 3000:
        matched_len = _find_prefix(PREVIOUS_COMPACTION_IDS, current_prompt_ids)
        if matched_len > 300:
            prompt_chunk = current_prompt_ids[matched_len:]
            _shift_cache_offset(COMPACTION_CACHE, matched_len)
            PREVIOUS_COMPACTION_IDS = current_prompt_ids
            return StreamingResponse(_bridge(prompt_chunk, max_tokens, request_id, False, COMPACTION_CACHE, total_prompt_len), media_type="text/event-stream")
        
        COMPACTION_CACHE = _make_fresh_cache()
        PREVIOUS_COMPACTION_IDS = current_prompt_ids
        return StreamingResponse(_bridge(current_prompt_ids, max_tokens, request_id, False, COMPACTION_CACHE, total_prompt_len), media_type="text/event-stream")
        
    # Обычный фоновый запрос (автотайтл сессии Goose)
    ephemeral_cache = _make_fresh_cache()
    return StreamingResponse(_bridge(current_prompt_ids, max_tokens, request_id, False, ephemeral_cache, total_prompt_len), media_type="text/event-stream")

def _handle_agent_scenarios(current_prompt_ids: list, total_prompt_len: int, request_id: str, max_tokens: int):
    """Управляет основным многовитоковым кэшем Агента."""
    global AGENT_CACHE, PREVIOUS_AGENT_IDS
    logger.info(f"POST /v1/chat/completions | Target: [AGENT] (ID: {request_id})")

    if PREVIOUS_AGENT_IDS:
        matched_tokens_len = _find_prefix(PREVIOUS_AGENT_IDS, current_prompt_ids)
        if matched_tokens_len > 300:
            prompt_ids_chunk = current_prompt_ids[matched_tokens_len:]
            # 🎯 ЗАЩИТА ОТ VALUEERROR: Если дельта пустая, откатываемся на 1 токен назад
            if len(prompt_ids_chunk) == 0:
                matched_tokens_len -= 1
                prompt_ids_chunk = [current_prompt_ids[-1]]
                
            _shift_cache_offset(AGENT_CACHE, matched_tokens_len)
            logger.info(f"🎯 [Cache AGENT Hit] Reused context: {C_GREEN}{matched_tokens_len}{C_RESET} tokens. Delta: {len(prompt_ids_chunk)}")
            PREVIOUS_AGENT_IDS = current_prompt_ids
            return StreamingResponse(_bridge(prompt_ids_chunk, max_tokens, request_id, True, AGENT_CACHE, total_prompt_len), media_type="text/event-stream")

    logger.info(f"🧹 [Cache AGENT Miss] Full evaluation required: {total_prompt_len} tokens.")
    AGENT_CACHE = _make_fresh_cache()
    PREVIOUS_AGENT_IDS = current_prompt_ids
    return StreamingResponse(_bridge(current_prompt_ids, max_tokens, request_id, True, AGENT_CACHE, total_prompt_len), media_type="text/event-stream")

# ------------------------------------------------------------
# 🛠️ СЛУЖЕБНЫЕ УТИЛИТЫ КЭШ-ДВИЖКА
# ------------------------------------------------------------
def _find_prefix(list1: list, list2: list) -> int:
    min_len = min(len(list1), len(list2))
    for i in range(min_len):
        if list1[i] != list2[i]: return i
    return min_len

def _make_fresh_cache():
    try:
        cache = mlx_lm.models.cache.make_prompt_cache(model)
        for layer in cache:
            if hasattr(layer, "bits"): layer.bits = 3
            if hasattr(layer, "quantized_start"): layer.quantized_start = 64
        return cache
    except Exception:
        return [mlx_lm.models.cache.KVCache() for _ in range(len(model.layers) if hasattr(model, "layers") else 32)]

def _shift_cache_offset(cache_registry: list, offset_val: int):
    for layer in cache_registry:
        if hasattr(layer, "offset"): layer.offset = offset_val
        elif hasattr(layer, "step"): layer.step = offset_val

async def _bridge(prompt_ids, max_tokens, r_id, has_tools, cache_obj, total_len):
    async for chunk in async_queue_bridge(model, tokenizer, prompt_ids, max_tokens, r_id, has_tools, args.prefill_step_size, cache_obj, args.model, total_len, None):
        yield chunk
    if ASYNC_SERVER_LOCK.locked(): ASYNC_SERVER_LOCK.release()

@app.get("/v1/context/status")
async def get_context_status():
    return {"total_prompt_len": len(PREVIOUS_AGENT_IDS)}

uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
