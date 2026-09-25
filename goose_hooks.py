import logging
import json

logger = logging.getLogger("goose_server.hooks")

def convert_openai_to_hf_tool(openai_tool: dict) -> dict:
    """Converts the tool description format from OpenAI to Hugging Face standard."""
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
    """Normalizes messages layout and strictly deactivates model reasoning layers."""
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

    # 🎯 Prompt-level User Intervention перед генерацией шагов 3/5 и 4/5 (при hit_count == 2 и 3)
    from anti_loop import anti_loop_engine, C_BOLD, C_CYAN, C_YELLOW, C_RESET
    if anti_loop_engine.hit_count in (2, 3) and anti_loop_engine.last_tool:
        step_label = "4/5" if anti_loop_engine.hit_count == 3 else "3/5"
        warning_tag = "USER DIRECTIVE - FINAL WARNING" if anti_loop_engine.hit_count == 3 else "USER INTERVENTION"
        user_intervention_text = (
            f"[{warning_tag}]: You are stuck calling '{anti_loop_engine.last_tool}' repeatedly with identical arguments. "
            f"As the human operator, I instruct you: do NOT retry this command. Change your approach, inspect different files, or ask me for clarification."
        )
        # Добавляем роль 'user' в конец контекста (вызывает сдвиг last_query_index в Jinja)
        fixed_messages.append({"role": "user", "content": user_intervention_text})
        logger.warning(
            f"{C_BOLD}{C_CYAN}👤 [USER INTERVENTION INJECTED - {step_label}]{C_RESET} {C_YELLOW}Повтор {step_label}: "
            f"Внедряем прямое указание оператора [{warning_tag}] в контекст модели. "
            f"(Примечание: роль user сдвигает last_query_index в Jinja, вызывая штатный сброс кэша из-за очистки устаревших <think> блоков).{C_RESET}"
        )

    # 🎯 Enforce reasoning suppression across all template layout arguments
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "thinking": False,
        "enable_thinking": False,
        "chat_template_args": {
            "enable_thinking": False,
            "thinking": False
        }
    }
    
    if tools is not None:
        template_kwargs["tools"] = [convert_openai_to_hf_tool(t) for t in tools if isinstance(t, dict)]

    return fixed_messages, template_kwargs
