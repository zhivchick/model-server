# response_formatters.py
import json
import time
import uuid

def build_streaming_chunk(request_id: str, model_name: str, content: str = None, tool_name: str = None, tool_args: str = None, finish_reason: str = None, prompt_len: int = 0, completion_len: int = 0) -> str:
    """Формирует чанк Server-Sent Events (SSE) с поддержкой OpenAI Usage для Goose."""
    delta = {}
    if tool_name:
        delta = {
            "role": "assistant",
            "tool_calls": [{
                "index": 0,
                "id": f"call_{request_id.split('-')[-1]}",
                "type": "function",
                "function": {"name": tool_name, "arguments": tool_args}
            }]
        }
    elif content is not None:
        delta = {"content": content}

    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    }

    # 🎯 ОЖИВЛЯЕМ ТРЕКЕР GOOSE: Если стрим завершается, принудительно прокидываем usage объект
    if finish_reason in ["stop", "tool_calls"] and prompt_len > 0:
        chunk["usage"] = {
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_len,
            "total_tokens": prompt_len + completion_len
        }

    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

def build_monolithic_response(text: str, model_name: str, tool_name: str = None, tool_args: str = None) -> dict:
    choices = []
    if tool_name:
        choices.append({
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": tool_args}
                }]
            },
            "finish_reason": "tool_calls"
        })
    else:
        choices.append({
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text.strip()
            },
            "finish_reason": "stop"
        })
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": choices
    }
