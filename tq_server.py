# tq_server.py
import sys
import uuid
import argparse
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import stream_generate

# Импортируем наши новые изолированные слои
from goose_hooks import apply_pre_call_hooks
from qwen_xml_parser import QwenXmlParser
from response_formatters import build_monolithic_response, build_streaming_chunk

parser = argparse.ArgumentParser(description="Modular MLX OpenAI Server")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--max-tokens", type=int, default=4096)
args, unknown = parser.parse_known_args()

print(f"📦 Загрузка модели Qwen 9B: {args.model}...")
model, tokenizer = mlx_lm.load(args.model)

print("\n" + "="*60)
print("🚀 [Modular Advanced Server] Успешно запущен!")
print("💡 Архитектура: Раздельные слои логики (Hooks, XML Parser, Formatters)")
print("💡 Сжатие: Нативный 3-bit KV-кэш Apple (kv_bits=3)")
print("="*60 + "\n")

app = FastAPI()

# --- СТРИМИНГОВЫЙ ГЕНЕРАТОР ДЛЯ GOOSE ---
async def async_stream_generator(prompt, max_tokens, request_id, has_tools):
    parser = QwenXmlParser()
    mem_before = mx.get_active_memory()
    
    print(f"\n🖥️  [DEBUG] --- СТАРТ ГЕНЕРАЦИИ (ID: {request_id}) ---")
    
    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, kv_bits=3, quantized_kv_start=64):
        # 1. Проверяем, не пошел ли вызов инструмента через наш XML-класс
        is_tool = parser.parse_chunk(response.text)
        
        if is_tool:
            # Не шлем Гусю недописанные куски XML! Ждем полной сборки тега в памяти воркера.
            continue
        else:
            # Обычный текстовый токен — выводим в консоль и мгновенно стримим в Goose
            sys.stdout.write(response.text)
            sys.stdout.flush()
            yield build_streaming_chunk(request_id, args.model, content=response.text)
            
        await asyncio.sleep(0.001)

    # Когда генерация завершилась, проверяем — был ли это вызов инструмента?
    if parser.in_tool_call:
        final_args = parser.extract_final_arguments()
        print(f"\n🎯 [DEBUG TOOL CALL]: Функция '{parser.tool_name}' -> Аргументы: {final_args}")
        # Шлем Гусю один синтаксически чистый, финальный JSON-пакет вызова инструмента
        yield build_streaming_chunk(request_id, args.model, tool_name=parser.tool_name, tool_args=final_args, finish_reason="tool_calls")
    else:
        # Обычное завершение текста
        yield build_streaming_chunk(request_id, args.model, finish_reason="stop")
        
    mem_after = mx.get_active_memory()
    print(f"\n📊 [DEBUG MEMORY] Пиковый кэш VRAM: {(mem_after - mem_before) / 1e9:.4f} GB")
    yield "data: [DONE]\n\n"


# --- ГЛАВНЫЙ ЭНДПОИНТ API ---
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    max_tokens = body.get("max_tokens", args.max_tokens)
    request_id = f"chatcmpl-{uuid.uuid4()}"
    
    # 1. Вызываем слой входящих хаков (чистим сообщения, отключаем размышления)
    fixed_messages, template_kwargs = apply_pre_call_hooks(body)
    has_tools = "tools" in template_kwargs

    # 2. Рендерим чистую строку промпта через токенизатор Apple
    prompt = tokenizer.apply_chat_template(fixed_messages, **template_kwargs)
    
    # 3. ПРОВЕРЯЕМ РЕЖИМ: Стриминг (Goose) или Монолит (curl)
    if body.get("stream", False):
        return StreamingResponse(
            async_stream_generator(prompt, max_tokens, request_id, has_tools),
            media_type="text/event-stream"
        )
    else:
        # Линейный запуск для классического curl-запроса
        print(f"\n🖥️  [DEBUG] --- МОНОЛИТНЫЙ ЗАПРОС (CURL) ---")
        text_accumulator = []
        parser = QwenXmlParser()
        
        for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, kv_bits=3, quantized_kv_start=64):
            parser.parse_chunk(response.text)
            if not parser.in_tool_call:
                text_accumulator.append(response.text)
                
        if parser.in_tool_call:
            response_json = build_monolithic_response("", args.model, tool_name=parser.tool_name, tool_args=parser.extract_final_arguments())
        else:
            response_json = build_monolithic_response("".join(text_accumulator), args.model)
            
        return JSONResponse(content=response_json)

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
