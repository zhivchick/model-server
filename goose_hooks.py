# goose_hooks.py
import logging

logger = logging.getLogger("goose_server.hooks")

def apply_pre_call_hooks(body: dict) -> tuple:
    """
    Перехватывает входящие данные, чистит историю сообщений от сложных структур,
    предотвращая ошибку Jinja2 'Can only get item pairs from a mapping'.
    Синхронизирует параметры отключения размышлений.
    """
    messages = body.get("messages", [])
    tools = body.get("tools", None)
    
    # 1. Жесткий фикс истории сообщений
    fixed_messages = []
    for msg in messages:
        clean_msg = msg.copy()
        # Если это отчет о работе инструмента от Гуся и контент пришел в виде сложного объекта/списка
        if clean_msg.get("role") == "tool" and not isinstance(clean_msg.get("content"), str):
            import json
            # Принудительно упаковываем его в плоскую строку, чтобы Jinja2 фильтр .items() не падал
            clean_msg["content"] = json.dumps(clean_msg.get("content"), ensure_ascii=False)
        fixed_messages.append(clean_msg)

    # 2. Формируем аргументы шаблонизатора (транслируем ваш chat_template_kwargs)
    client_kwargs = body.get("chat_template_kwargs", {})
    template_kwargs = {
        "enable_thinking": False,  # Наш дефолт
        "tokenize": False,
        "add_generation_prompt": True
    }
    # Накатываем параметры, прилетевшие от клиента/плагина LiteLLM
    template_kwargs.update(client_kwargs)
    
    if tools is not None:
        template_kwargs["tools"] = tools
        logger.info(f"🔌 [HOOK] Найдено {len(tools)} инструментов от Goose. Передаем в контекст.")

    return fixed_messages, template_kwargs
