import os
import gc
import mlx.core as mx
import uuid
import argparse
import logging
import datetime
import re
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

import mlx_lm
from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache, trim_prompt_cache
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
parser.add_argument("--prefill-step-size", type=int, default=1024)
parser.add_argument("--log-level", type=str, default="info", choices=["info", "debug", "warning", "error"])
parser.add_argument(
    "--anti-loop-free-limit", "--free-loop-limit",
    type=int,
    default=int(os.environ.get("ANTI_LOOP_FREE_HITS", "5")),
    help="Number of free repetitions allowed before anti-loop escalation (default: 5)"
)
parser.add_argument(
    "--repetition-penalty",
    type=float,
    default=1.0,
    help="Penalty for repeating tokens (default: 1.0 - disabled to preserve code/tool syntax)"
)
parser.add_argument(
    "--kv-bits",
    type=int,
    default=8,
    choices=[0, 4, 8],
    help="KV cache quantization bits (0 = disable quantization, 8 = 8-bit, 4 = 4-bit)"
)
parser.add_argument(
    "--kv-group-size",
    type=int,
    default=64,
    help="Group size for KV cache quantization (default: 64)"
)
parser.add_argument(
    "--quantized-kv-start",
    type=int,
    default=32768,
    help="Token threshold where KV cache quantization kicks in (default: 32768 for fast FP16 up to 32k)"
)
args, unknown = parser.parse_known_args()

numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("mlx_lm_server")

from anti_loop import anti_loop_engine
anti_loop_engine.configure(args.anti_loop_free_limit)

C_GREEN, C_YELLOW, C_CYAN, C_RESET = "\033[92m", "\033[93m", "\033[96m", "\033[0m"

print(f"📦 Loading model: {args.model}...")
model, tokenizer = mlx_lm.load(args.model)

if args.kv_bits > 0:
    logger.info(
        f"{C_CYAN}⚙️ [KV-CACHE CONFIG] Quantization: {args.kv_bits}-bit "
        f"(group_size={args.kv_group_size}, starts after {args.quantized_kv_start} tokens){C_RESET}"
    )
else:
    logger.info(f"{C_CYAN}⚙️ [KV-CACHE CONFIG] Quantization disabled (native FP16/BF16 cache){C_RESET}")

# 🛡️ ПАТЧ JINJA-ШАБЛОНА QWEN 3.5:
# Защита от сброса кэша при изменении ns.last_query_index из-за появления role="user" / turn-context.
if hasattr(tokenizer, "chat_template") and isinstance(tokenizer.chat_template, str):
    if "ns.last_query_index" in tokenizer.chat_template:
        tokenizer.chat_template = re.sub(
            r'loop\.index0\s*>\s*ns\.last_query_index',
            'true',
            tokenizer.chat_template
        )
        logger.info(f"{C_GREEN}🛡️ [JINJA PATCH] Заблокирован сброс <think> по ns.last_query_index в chat_template.{C_RESET}")

AGENT_CACHE = None
PREVIOUS_AGENT_IDS = []
COMPACTION_CACHE = None
PREVIOUS_COMPACTION_IDS = []
LAST_LIVE_RAW_TEXT = ""
ASYNC_SERVER_LOCK = asyncio.Lock()

app = FastAPI()

# ------------------------------------------------------------
# 🌐 ОСНОВНОЙ СЕТЕВОЙ ЭНДПОИНТ
# ------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    await ASYNC_SERVER_LOCK.acquire()
    try:
        body = await request.json()
        request_id = f"chatcmpl-{uuid.uuid4()}"
        
        fixed_messages, template_kwargs = apply_pre_call_hooks(body)
        is_agent = body.get("tools") is not None

        fixed_messages = _apply_hardware_time_lock(fixed_messages)
        current_prompt_ids, total_prompt_len = _render_and_tokenize(fixed_messages, template_kwargs)
        _print_pipeline_telemetry(request_id, is_agent, total_prompt_len)

        max_tok = body.get("max_tokens", args.max_tokens)
        if not is_agent:
            return _handle_utility_scenarios(current_prompt_ids, total_prompt_len, request_id, max_tok)
        else:
            return _handle_agent_scenarios(current_prompt_ids, total_prompt_len, request_id, max_tok)

    except Exception as e:
        if ASYNC_SERVER_LOCK.locked():
            ASYNC_SERVER_LOCK.release()
        raise e

# ------------------------------------------------------------
# 🧠 ИЗОЛИРОВАННЫЕ ПРОЦЕДУРЫ ОБРАБОТКИ КОНТЕКСТА
# ------------------------------------------------------------
def _apply_hardware_time_lock(messages: list) -> list:
    for msg in messages:
        if "content" in msg and isinstance(msg["content"], str):
            content_str = msg["content"]
            content_str = re.sub(r"<current-time>.*?</current-time>", "<current-time>STATIC_TIME_LOCK</current-time>", content_str)
            content_str = re.sub(r"<compaction>.*?</compaction>", "<compaction>STATIC_COMPACTION_LOCK</compaction>", content_str)
            msg["content"] = content_str
    return messages

def _render_and_tokenize(messages: list, template_kwargs: dict) -> tuple:
    global LAST_LIVE_RAW_TEXT
    full_prompt_string = tokenizer.apply_chat_template(messages, **template_kwargs)
    LAST_LIVE_RAW_TEXT = full_prompt_string
    current_prompt_ids = tokenizer.encode(full_prompt_string, add_special_tokens=False)
    return current_prompt_ids, len(current_prompt_ids)

def _print_pipeline_telemetry(request_id: str, is_agent: bool, total_prompt_len: int):
    logger.info(f"📍 [PIPELINE] req={request_id} target={'AGENT' if is_agent else 'UTILITY'} prompt_len={total_prompt_len}")

def _handle_utility_scenarios(current_prompt_ids: list, total_prompt_len: int, request_id: str, max_tokens: int):
    global AGENT_CACHE, PREVIOUS_AGENT_IDS
    logger.info(f"POST /v1/chat/completions | Target: [UTILITY] (ID: {request_id}, len={total_prompt_len})")
    
    # 🚨 ЗАЩИТА ОТ OOM НА КОМПАКТЕ (48k+ токенов):
    # Если прилетел тяжелый запрос без тулов (> 3000 токенов) — это Compaction истории в Goose.
    # После компакта старая история агента все равно сбрасывается, поэтому мы ОБЯЗАНЫ
    # освободить AGENT_CACHE ДО начала префилла компакта, чтобы не держать два кэша по 48k в 16GB RAM!
    if total_prompt_len > 3000:
        if AGENT_CACHE is not None:
            logger.warning(
                f"{C_YELLOW}🧹 [PRE-COMPACT VRAM EVICTION] Освобождаем AGENT_CACHE ({len(PREVIOUS_AGENT_IDS)} токенов) "
                f"перед запуском тяжелого компакта ({total_prompt_len} токенов), чтобы избежать Metal OOM!{C_RESET}"
            )
            AGENT_CACHE = None
            PREVIOUS_AGENT_IDS = []
            gc.collect()
            mx.clear_cache()

        ephemeral_cache = _make_fresh_cache()
        # Для компакта принудительно жмем кэш в 4 бита сразу с 512 токенов (экономия >2.5 ГБ VRAM)
        return StreamingResponse(
            _bridge(
                current_prompt_ids,
                max_tokens,
                request_id,
                False,
                ephemeral_cache,
                total_prompt_len,
                None,
                override_kv_bits=4,
                override_kv_start=512
            ),
            media_type="text/event-stream"
        )
        
    # Короткий фоновый запрос (например, генерация заголовка сессии на ~200 токенов):
    # здесь AGENT_CACHE трогать НЕЛЬЗЯ, так как агент продолжит работу сразу после автотайтла!
    ephemeral_cache = _make_fresh_cache()
    return StreamingResponse(
        _bridge(current_prompt_ids, max_tokens, request_id, False, ephemeral_cache, total_prompt_len, None),
        media_type="text/event-stream"
    )


def _handle_agent_scenarios(current_prompt_ids: list, total_prompt_len: int, request_id: str, max_tokens: int):
    global AGENT_CACHE, PREVIOUS_AGENT_IDS
    logger.info(f"POST /v1/chat/completions | Target: [AGENT] (ID: {request_id})")

    if AGENT_CACHE is not None and PREVIOUS_AGENT_IDS:
        matched_tokens_len = _find_prefix(PREVIOUS_AGENT_IDS, current_prompt_ids)

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
            except Exception:
                pass

        ok, effective_matched_len = _try_align_cache(AGENT_CACHE, matched_tokens_len)
        if effective_matched_len > 300 and ok:
            prompt_ids_chunk = current_prompt_ids[effective_matched_len:]
            if len(prompt_ids_chunk) == 0:
                if can_trim_prompt_cache(AGENT_CACHE):
                    trim_prompt_cache(AGENT_CACHE, 1)
                    effective_matched_len -= 1
                    prompt_ids_chunk = [current_prompt_ids[-1]]
                else:
                    AGENT_CACHE = _make_fresh_cache()
                    PREVIOUS_AGENT_IDS = list(current_prompt_ids)
                    return StreamingResponse(
                        _bridge(current_prompt_ids, max_tokens, request_id, True, AGENT_CACHE, total_prompt_len, None),
                        media_type="text/event-stream"
                    )
                
            logger.info(f"🎯 [Cache AGENT Hit] Reused context: {C_GREEN}{effective_matched_len}{C_RESET} tokens. Delta: {len(prompt_ids_chunk)}")
            PREVIOUS_AGENT_IDS = list(current_prompt_ids)
            return StreamingResponse(
                _bridge(prompt_ids_chunk, max_tokens, request_id, True, AGENT_CACHE, total_prompt_len, None),
                media_type="text/event-stream"
            )

    logger.info(f"🧹 [Cache AGENT Miss] Full evaluation required: {total_prompt_len} tokens.")
    AGENT_CACHE = _make_fresh_cache()
    PREVIOUS_AGENT_IDS = list(current_prompt_ids)
    return StreamingResponse(
        _bridge(current_prompt_ids, max_tokens, request_id, True, AGENT_CACHE, total_prompt_len, None),
        media_type="text/event-stream"
    )

# ------------------------------------------------------------
# 🛠️ СЛУЖЕБНЫЕ УТИЛИТЫ КЭШ-ДВИЖКА
# ------------------------------------------------------------
def _find_prefix(list1: list, list2: list) -> int:
    min_len = min(len(list1), len(list2))
    for i in range(min_len):
        if list1[i] != list2[i]:
            return i
    return min_len

def _make_fresh_cache():
    return make_prompt_cache(model)

def _get_cache_offset(cache_registry: list) -> int:
    for layer in cache_registry:
        offset = getattr(layer, "offset", None)
        if offset is not None:
            return offset
    return 0

def _try_align_cache(cache_registry: list, matched_tokens_len: int) -> tuple[bool, int]:
    """
    Безопасно подрезает сгенерированный хвост в кэше до границы прошлого промпта.
    """
    current_offset = _get_cache_offset(cache_registry)
    if current_offset <= 0:
        return False, 0

    effective_target = min(matched_tokens_len, current_offset)
    tokens_to_trim = current_offset - effective_target

    if tokens_to_trim == 0:
        return True, effective_target

    for layer in cache_registry:
        if hasattr(layer, "is_trimmable") and layer.is_trimmable():
            layer.trim(tokens_to_trim)
        elif hasattr(layer, "offset") and layer.offset is not None:
            layer.offset = effective_target

    logger.info(
        f"✂️ [CACHE ALIGN] Trimmed {tokens_to_trim} tail tokens "
        f"(cache_offset: {current_offset} -> {effective_target})"
    )
    return True, effective_target

async def _bridge(
    prompt_ids,
    max_tokens,
    r_id,
    has_tools,
    cache_obj,
    total_len,
    on_complete_cb,
    override_kv_bits=None,
    override_kv_start=None
):
    effective_kv_bits = override_kv_bits if override_kv_bits is not None else args.kv_bits
    effective_kv_start = override_kv_start if override_kv_start is not None else args.quantized_kv_start
    try:
        async for chunk in async_queue_bridge(
            model,
            tokenizer,
            prompt_ids,
            max_tokens,
            r_id,
            has_tools,
            args.prefill_step_size,
            cache_obj,
            args.model,
            total_len,
            args.repetition_penalty,
            effective_kv_bits,
            args.kv_group_size,
            effective_kv_start,
            on_complete_cb
        ):
            yield chunk
    finally:
        # Если это был одноразовый утилитный запрос — сразу освобождаем его тензоры и кэш Metal
        if not has_tools:
            del cache_obj
            gc.collect()
            mx.clear_cache()
        if ASYNC_SERVER_LOCK.locked():
            ASYNC_SERVER_LOCK.release()


@app.get("/v1/context/status")
async def get_context_status():
    global PREVIOUS_AGENT_IDS
    current_len = len(PREVIOUS_AGENT_IDS)
    logger.info(f"📊 {C_YELLOW}[HOOK API REQUEST]{C_RESET} External watchdog checked context size. Current: {C_GREEN}{current_len}{C_RESET} tokens.")
    return {"total_prompt_len": current_len}

@app.get("/v1/context/raw")
async def get_raw_context():
    global PREVIOUS_AGENT_IDS, LAST_LIVE_RAW_TEXT
    return {
        "total_tokens_len": len(PREVIOUS_AGENT_IDS),
        "raw_text": LAST_LIVE_RAW_TEXT or ""
    }

uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
