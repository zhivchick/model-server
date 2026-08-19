# goose_hooks.py
import logging
import json

logger = logging.getLogger("goose_server.hooks")

def convert_openai_to_hf_tool(openai_tool: dict) -> dict:
    """Конвертирует формат описания инструментов в формат Hugging Face."""
    if not isinstance(openai_tool, dict) or "function" not in openai_tool:
        return openai_tool
    func = openai_tool["function"]
    openai_params = func.get("parameters", {})
    hf_tool = {
        "type": "function",
        "function": {
            "name": func.get("name"),
            "description": func.get("description", ""),
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
    if isinstance(openai_params, dict):
        properties = openai_params.get("properties", {})
        required = openai_params.get("required", [])
        hf_tool["function"]["parameters"]["properties"] = properties if isinstance(properties, dict) else {}
        hf_tool["function"]["parameters"]["required"] = required if isinstance(required, list) else []
    return hf_tool

def apply_pre_call_hooks(body: dict) -> tuple:
    """Нормализует сообщения и инструменты под правила Hugging Face."""
    messages = body.get("messages", [])
    tools = body.get("tools", None)
    
    fixed_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        clean_msg = msg.copy()
        role = clean_msg.get("role")
        
        if clean_msg.get("content") is None:
            clean_msg["content"] = ""
            
        if role == "tool" and "content" in clean_msg:
            content = clean_msg["content"]
            if not isinstance(content, str):
                clean_msg["content"] = json.dumps(content, ensure_ascii=False)
                
        elif role == "assistant" and "tool_calls" in clean_msg:
            openai_calls = clean_msg["tool_calls"]
            if isinstance(openai_calls, list):
                hf_calls = []
                for call in openai_calls:
                    if isinstance(call, dict) and "function" in call:
                        func_part = call["function"]
                        f_name = func_part.get("name")
                        f_args = func_part.get("arguments", "{}")
                        if isinstance(f_args, str):
                            try: f_args = json.loads(f_args)
                            except Exception: f_args = {"raw_args": f_args}
                        hf_calls.append({"name": f_name, "arguments": f_args if isinstance(f_args, dict) else {}})
                clean_msg["tool_calls"] = hf_calls
        fixed_messages.append(clean_msg)

    # Защита от ошибки No user query found в Jinja Qwen
    if fixed_messages and fixed_messages[-1].get("role") in ["tool", "assistant"]:
        has_user = any(m.get("role") == "user" for m in fixed_messages)
        if not has_user or fixed_messages[-1].get("role") == "tool":
            fixed_messages.append({"role": "user", "content": "Continue and execute the next tool step based on the output above."})

    fixed_tools = [convert_openai_to_hf_tool(t) for t in tools if isinstance(t, dict)] if tools is not None else None

    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "chat_template_args": {"enable_thinking": False}
    }
    if fixed_tools:
        template_kwargs["tools"] = fixed_tools

    return fixed_messages, template_kwargs
