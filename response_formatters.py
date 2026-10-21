import json
import time

def build_streaming_chunk(
    request_id: str,
    model_name: str,
    content: str = None,
    tool_name: str = None,
    tool_args: str = None,
    finish_reason: str = None,
    prompt_len: int = 0,
    completion_len: int = 0
) -> str:
    """Constructs a Server-Sent Events (SSE) chunk containing OpenAI Usage metadata for Goose tracking."""
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

    if finish_reason in ["stop", "tool_calls"] and prompt_len > 0:
        chunk["usage"] = {
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_len,
            "total_tokens": prompt_len + completion_len
        }

    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
