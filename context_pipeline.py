import re
import json
import logging

logger = logging.getLogger("mlx_lm_server.pipeline")

C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_RED    = "\033[91m"
C_YELLOW = "\033[93m"

def apply_hardware_time_lock(fixed_messages: list) -> list:
    """
    🎯 СТАБИЛЬНЫЙ ЗАМОК ВРЕМЕНИ.
    Замораживает теги времени строго до вызова токенизатора, предотвращая сдвиг KV-cache.
    """
    for msg in fixed_messages:
        if "content" in msg and isinstance(msg["content"], str):
            content_str = msg["content"]
            content_str = re.sub(r"<current-time>.*?</current-time>", "<current-time>STATIC_TIME_LOCK</current-time>", content_str)
            content_str = re.sub(r"<compaction>.*?</compaction>", "<compaction>STATIC_COMPACTION_LOCK</compaction>", content_str)
            msg["content"] = content_str
    return fixed_messages

def dump_nuclear_prompt_trace(full_prompt_string: str, PREVIOUS_AGENT_IDS: list, request_id: str):
    """
    💾 ЯДЕРНЫЙ ЛОГГЕР ПРОМПТОВ.
    Записывает слепки промптов на диск для побайтного анализа структуры.
    """
    step_type = "STEP_2_HIT" if PREVIOUS_AGENT_IDS else "STEP_1_MISS"
    debug_filename = f"debug_prompt_{step_type}_{request_id[-6:]}.txt"
    try:
        with open(debug_filename, "w", encoding="utf-8") as f:
            f.write(full_prompt_string)
        logger.warning(f"💾 [NUCLEAR TRACE] Prompt snapshot dumped: {debug_filename}")
    except Exception:
        pass
