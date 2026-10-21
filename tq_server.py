import os
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
# 🎯 Дефолт 1.0: не ломаем синтаксис кода и XML-тегов штрафами
parser.add_argument(
    "--repetition-penalty",
    type=float,
    default=1.0,
    help="Penalty for repeating tokens (default: 1.0 - disabled to preserve code/tool syntax)"
)
# 🎯 Нативное квантование KV-кэша через mlx_lm (8-bit по дефолту = ~2.1 GB на 64k контекста у Qwen 3.5)
parser.add_argument(
    "--kv-bits",
    type=int,
    default=8,
    choices=[0, 4, 8],
    help="KV cache quantization bits (8 = optimal for 64k+ context, 4 = extreme compression, 0 = FP16)"
)
parser.add_argument("--kv-group-size", type=int, default=64)
parser.add_argument("--quantized-kv-start", type=int, default=1024)
args, unknown = parser.parse_known_args()

numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("mlx_lm_server")

from anti_loop import anti_loop_engine
anti_loop_engine.configure(args.anti_loop_free_limit)

C_GREEN, C_YELLOW, C_RESET = "\033[92m", "\033[93m", "\033[0m"

print(f"📦 Loading model: {args.model}...")
model, tokenizer = mlx_lm.load(args.model)

# 🎯 Без двойной аллокации: инициализируем лениво через make_prompt_cache(model)
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
    global LAST_LIVE_RAW_TEXT
    full_prompt_string = tokenizer.apply_chat_template(messages, **template_kwargs)
    LAST_LIVE_RAW_TEXT = full_prompt_string
    current_prompt_ids = tokenizer.encode(full_prompt_string, add_special_tokens=False)
    return current_prompt_ids, len(current_prompt_ids)

def _print_pipeline_telemetry(request_id: str, is_agent: bool, total_prompt_len: int):
    logger.info(f"📍 [PIPELINE] req={request_id} target={'AGENT' if is_agent else 'UTILITY'} prompt_len={total_prompt_len}")

# ------------------------------------------------------------
# ⚡ ИЗОЛИРОВАННЫЕ СЦЕНАРИИ КЭШИРОВАНИЯ И ИНФЕРЕНСА
# ------------------------------------------------------------
def _append_agent_generated_tokens(gen_ids: list):
    """Сохраняет сгенерированные токены в PREVIOUS_AGENT_IDS (кроме финального <|im_end|>),
    чтобы следующий запрос идеально продолжал кэш без отката и пересчета ответа ассистента."""
    global PREVIOUS_AGENT_IDS
    # В шаблоне Qwen ответ заканчивается на <|im_end|>, после которого в следующем запросе идет \n.
    # Если последний токен — это eos/im_end, оставляем его в истории: в новом промпте он стоит на том же месте!
    PREVIOUS_AGENT_IDS.extend(gen_ids)

def _append_compaction_generated_tokens(gen_ids: list):
    global PREVIOUS_COMPACTION_IDS
    PREVIOUS_COMPACTION_IDS.extend(gen_ids)

def _handle_utility_scenarios(current_prompt_ids: list, total_prompt_len: int, request_id: str, max_tokens: int):
    global COMPACTION_CACHE, PREVIOUS_COMPACTION_IDS
    logger.info(f"POST /v1/chat/completions | Target: [UTILITY] (ID: {request_id})")
    
    if total_prompt_len > 3000:
        if COMPACTION_CACHE is not None and PREVIOUS_COMPACTION_IDS:
            matched_len = _find_prefix(PREVIOUS_COMPACTION_IDS, current_prompt_ids)
            if matched_len > 300 and _try_align_cache(COMPACTION_CACHE, matched_len):
                prompt_chunk = current_prompt_ids[matched_len:]
                if len(prompt_chunk) == 0:
                    if can_trim_prompt_cache(COMPACTION_CACHE):
                        trim_prompt_cache(COMPACTION_CACHE, 1)
                        matched_len -= 1
                        prompt_chunk = [current_prompt_ids[-1]]
                PREVIOUS_COMPACTION_IDS = list(current_prompt_ids)
                return StreamingResponse(
                    _bridge(prompt_chunk, max_tokens, request_id, False, COMPACTION_CACHE, total_prompt_len, _append_compaction_generated_tokens),
                    media_type="text/event-stream"
                )
        
        COMPACTION_CACHE = _make_fresh_cache()
        PREVIOUS_COMPACTION_IDS = list(current_prompt_ids)
        return StreamingResponse(
            _bridge(current_prompt_ids, max_tokens, request_id, False, COMPACTION_CACHE, total_prompt_len, _append_compaction_generated_tokens),
            media_type="text/event-stream"
        )
        
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

        if matched_tokens_len > 300 and _try_align_cache(AGENT_CACHE, matched_tokens_len):
            prompt_ids_chunk = current_prompt_ids[matched_tokens_len:]
            if len(prompt_ids_chunk) == 0:
                if can_trim_prompt_cache(AGENT_CACHE):
                    trim_prompt_cache(AGENT_CACHE, 1)
                    matched_tokens_len -= 1
                    prompt_ids_chunk = [current_prompt_ids[-1]]
                else:
                    # Для нетранкабельных гибридных кэшей (DeltaNet) при нулевой дельте пересоздаем кэш
                    AGENT_CACHE = _make_fresh_cache()
                    PREVIOUS_AGENT_IDS = list(current_prompt_ids)
                    return StreamingResponse(
                        _bridge(current_prompt_ids, max_tokens, request_id, True, AGENT_CACHE, total_prompt_len, _append_agent_generated_tokens),
                        media_type="text/event-stream"
                    )
                
            logger.info(f"🎯 [Cache AGENT Hit] Reused context: {C_GREEN}{matched_tokens_len}{C_RESET} tokens. Delta: {len(prompt_ids_chunk)}")
            PREVIOUS_AGENT_IDS = list(current_prompt_ids)
            return StreamingResponse(
                _bridge(prompt_ids_chunk, max_tokens, request_id, True, AGENT_CACHE, total_prompt_len, _append_agent_generated_tokens),
                media_type="text/event-stream"
            )

    logger.info(f"🧹 [Cache AGENT Miss] Full evaluation required: {total_prompt_len} tokens.")
    AGENT_CACHE = _make_fresh_cache()
    PREVIOUS_AGENT_IDS = list(current_prompt_ids)
    return StreamingResponse(
        _bridge(current_prompt_ids, max_tokens, request_id, True, AGENT_CACHE, total_prompt_len, _append_agent_generated_tokens),
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
    """Создает правильную структуру кэша под конкретную архитектуру модели (включая гибридный Qwen 3.5)."""
    return make_prompt_cache(model)

def _get_cache_offset(cache_registry: list) -> int:
    """Возвращает текущий offset по первому KVCache-слою."""
    for layer in cache_registry:
        offset = getattr(layer, "offset", None)
        if offset is not None:
            return offset
    return 0

def _try_align_cache(cache_registry: list, target_offset: int) -> bool:
    """
    Аккуратно выравнивает кэш по target_offset через штатный API mlx_lm без разрушения pre-allocated буферов.
    Возвращает True, если кэш готов к продолжению с target_offset, или False, если нужен полный сброс (например,
    когда требуется откат назад на гибридной модели с нетранкабельными слоями ArraysCache/DeltaNet).
    """
    current_offset = _get_cache_offset(cache_registry)
    if current_offset == target_offset:
        return True

    if current_offset > target_offset:
        tokens_to_trim = current_offset - target_offset
        if can_trim_prompt_cache(cache_registry):
            trim_prompt_cache(cache_registry, tokens_to_trim)
            return True
        else:
            # В гибридной модели Qwen 3.5 есть 24 слоя ArraysCache (Gated DeltaNet), которые нельзя откатить назад,
            # если расхождение произошло внутри уже сгенерированного куска (например, из-за стоп-слова или правки).
            # Если расхождение небольшое (до 8 токенов — например, разница в закрывающем токене после перехвата XML),
            # подрезаем только классические KVCache-слои, где хранятся позиционные ключи.
            if tokens_to_trim <= 8:
                for layer in cache_registry:
                    if hasattr(layer, "is_trimmable") and layer.is_trimmable():
                        layer.trim(tokens_to_trim)
                return True
            logger.warning(
                f"{C_YELLOW}⚠️ [HYBRID CACHE RESET] Нужно откатить {tokens_to_trim} токенов, но модель содержит рекуррентные слои DeltaNet. Пересоздаем кэш.{C_RESET}"
            )
            return False

    # Если current_offset < target_offset — кэш почему-то отстает от истории, безопаснее пересчитать
    return False

async def _bridge(prompt_ids, max_tokens, r_id, has_tools, cache_obj, total_len, on_complete_cb):
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
            args.kv_bits,
            args.kv_group_size,
            args.quantized_kv_start,
            on_complete_cb
        ):
            yield chunk
    finally:
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
