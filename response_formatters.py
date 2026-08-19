# response_formatters.py
import json
import time

def build_monolithic_response(text: str, model_name: str, tool_name: str = None, tool_args: str = None) -> dict:
    """Формирует стандартный монолитный JSON-ответ OpenAI API для обычного curl-запроса."""
    choices = []
    if tool_name:
        choices.append({
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{uuid_to_id()}",
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
        "id": f"chatcmpl-{uuid_to_id()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": choices
    }

def build_streaming_chunk(request_id: str, model_name: str, content: str = None, tool_name: str = None, tool_args: str = None, finish_reason: str = None) -> str:
    """Формирует синтаксически чистый чанк в формате Server-Sent Events (SSE) для Goose."""
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
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

def uuid_to_id():
    import uuid
    return uuid.uuid4().hex[:24]
