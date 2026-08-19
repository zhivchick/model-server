# tq_server.py
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

parser = argparse.ArgumentParser(description="Stateful MLX OpenAI Server")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--max-tokens", type=int, default=4096)
parser.add_argument("--prefill-step-size", type=int, default=512)
args, unknown = parser.parse_known_args()

print(f"📦 Загрузка модели Qwen 9B: {args.model}...")
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

GLOBAL_PROMPT_CACHE = make_persistent_cache()
PREVIOUS_PROMPT_STRING = ""

print("\n" + "="*60 + "\n🚀 [Goose Stateful Server] Успешно запущен!\n💡 Архитектура: Модульные слои + Алгоритм Общего Префикса LCP\n" + "="*60 + "\n")
app = FastAPI()

def find_longest_common_prefix(s1: str, str2: str) -> int:
    """Возвращает длину наибольшего общего префикса между двумя строками."""
    min_len = min(len(s1), len(str2))
    for i in range(min_len):
        if s1[i] != str2[i]:
            return i
    return min_len

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global GLOBAL_PROMPT_CACHE, PREVIOUS_PROMPT_STRING
    body = await request.json()
    max_tokens = body.get("max_tokens", args.max_tokens)
    request_id = f"chatcmpl-{uuid.uuid4()}"
    
    logger.info(f"POST /v1/chat/completions")
    fixed_messages, template_kwargs = apply_pre_call_hooks(body)
    has_tools = body.get("tools") is not None

    # Генерируем полную текстовую строку промпта для текущего шага
    full_prompt_string = tokenizer.apply_chat_template(fixed_messages, **template_kwargs)
    
    # 🎯 ВЫЧИСЛЯЕМ МАКСИМАЛЬНОЕ СОВПАДЕНИЕ ТЕКСТА (LCP АЛГОРИТМ):
    if PREVIOUS_PROMPT_STRING:
        prefix_len = find_longest_common_prefix(PREVIOUS_PROMPT_STRING, full_prompt_string)
        
        # Если совпало больше 1000 символов (это гарантированно наша история диалога)
        if prefix_len > 1000:
            # Извлекаем кусок текста, который совпал символ-в-символ
            matched_text = full_prompt_string[:prefix_len]
            # Извлекаем только свежий хвост, который изменился или добавился
            new_text_chunk = full_prompt_string[prefix_len:]
            
            # Переводим в токены оба куска, чтобы узнать точное смещение для Си-ядра Apple
            matched_tokens_len = len(tokenizer.encode(matched_text))
            prompt_ids_chunk = tokenizer.encode(new_text_chunk)
            
            # Сдвигаем счетчик позиций внутри слоев глобального кэша Metal VRAM.
            # Мы принудительно обрезаем кэш до длины совпавших токенов!
            for layer_cache in GLOBAL_PROMPT_CACHE:
                if hasattr(layer_cache, "offset"):
                    layer_cache.offset = matched_tokens_len
            
            logger.info(f"🎯 [Cache LCP Hit] Совпало {matched_tokens_len} токенов истории! Сдвигаем offset.")
            logger.info(f"Evaluating ONLY incremental diff: {len(prompt_ids_chunk)} tokens.")
            
            PREVIOUS_PROMPT_STRING = full_prompt_string
            return StreamingResponse(
                async_queue_bridge(model, tokenizer, prompt_ids_chunk, max_tokens, request_id, has_tools, args.prefill_step_size, GLOBAL_PROMPT_CACHE, args.model),
                media_type="text/event-stream"
            )

    # Fallback: Если это старт новой сессии Goose
    logger.info("🧹 [Cache Miss] Новая сессия. Инициализируем полный префилл.")
    GLOBAL_PROMPT_CACHE = make_persistent_cache()
    PREVIOUS_PROMPT_STRING = full_prompt_string
    full_prompt_ids = tokenizer.encode(full_prompt_string)
    
    return StreamingResponse(
        async_queue_bridge(model, tokenizer, full_prompt_ids, max_tokens, request_id, has_tools, args.prefill_step_size, GLOBAL_PROMPT_CACHE, args.model),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
