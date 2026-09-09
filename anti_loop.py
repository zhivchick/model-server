import re
import json
import logging

logger = logging.getLogger("mlx_lm_server.antiloop")

C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_RED    = "\033[91m"
C_YELLOW = "\033[93m"
C_GREEN  = "\033[92m" # <--- А, она объявлена!

class AntiLoopEngine:
    def __init__(self):
        self.last_tool = None
        self.last_skeleton = None
        self.hit_count = 0

    def _build_argument_skeleton(self, args_dict: dict) -> str:
        raw_str = json.dumps(args_dict, ensure_ascii=False)
        return re.sub(r'\d+', '', raw_str)

    def evaluate_and_process(self, full_response_text: str, parser_tool_name: str, extracted_args_dict: dict) -> tuple:
        """
        🛡️ РАДАР СБОЕВ. Вызывается строго в конце генерации (stream_bridge.py).
        На витках 1 и 2 принудительно подменяет инструмент на echo && exit 1 для Goose.
        """
        func_match = re.search(r'<?function=([^>]+)>?', full_response_text)
        if func_match:
            tool_name = func_match.group(1).strip().replace('"', '').replace("'", "")
        else:
            tool_name = parser_tool_name if parser_tool_name else "shell"

        current_skeleton = self._build_argument_skeleton(extracted_args_dict)
        final_json_args = json.dumps(extracted_args_dict, ensure_ascii=False)

        is_edit_tool = tool_name in ["edit", "developer__edit", "patch"]

        # ХАК №2: ДЕТЕКТОР ИДЕНТИЧНЫХ ПРАВОК (DEPTH 0)
        if is_edit_tool:
            before_str = extracted_args_dict.get("before", "")
            after_str = extracted_args_dict.get("after", "")
            if before_str and after_str and before_str.strip() == after_str.strip():
                logger.error(f"{C_BOLD}{C_RED}🚨 [EDIT IDLE DETECTED] Identical before/after blocks.{C_RESET}")
                forced_tool_name = "shell"
                idle_warning = "echo 'Execution Error: The \"before\" and \"after\" parameters are byte-for-byte identical. Your edit action did NOT change any code.' && exit 1"
                return forced_tool_name, json.dumps({"command": idle_warning}, ensure_ascii=False)

        # Проверка на циклы
        if self.last_tool == tool_name and self.last_skeleton == current_skeleton:
            self.hit_count += 1
            logger.warning(f"{C_YELLOW}⚠️ [ANTI-LOOP FIREWALL] Repetitive pattern found! Tool: '{tool_name}', Depth: {self.hit_count}{C_RESET}")
            
            # ПОПЫТКА 6 (hit_count >= 5) -> ЖЕСТКИЙ ВЫЛЕТ НА ЮЗЕРА (БРЕЙК СЕССИИ)
            if self.hit_count >= 5:
                logger.error(f"{C_BOLD}{C_RED}🚨 [CONTEXT EMERGENCY] Hard threshold reached. Forcing Dialogue Brake.{C_RESET}")
                self.hit_count = 0
                self.last_tool = None
                self.last_skeleton = None
                return None, "Dialogue Brake: Repetition threshold exceeded. Control returned to human."

            # 🔥 ВОТ ОНО! ПОДМЕНА ОТВЕТА ТУЛА НА РАННИХ ЭТАПАХ (Depth 1 и 2) ПРЯМО В GOOSE!
            if self.hit_count == 1 or self.hit_count == 2:
                forced_tool_name = "shell"
                if is_edit_tool:
                    logger.error(f"{C_GREEN}💡 [SMART ASSIST] Continuous ambiguous edit detected! Injecting fake shell error into Goose.{C_RESET}")
                    if self.hit_count == 1:
                        payload_text = "echo 'Execution Error: Multiple matches found or code state unchanged. Do not retry the exact same string. Expand your context lines.' && exit 1"
                    else:
                        payload_text = "echo 'Execution Error: Ambiguity loop sustained. Rewrite your edit call by including 2-3 lines of surrounding code ABOVE and BELOW the target change.' && exit 1"
                else:
                    logger.error(f"{C_YELLOW}🚨 [FIREWALL BRICKWALL] Continuous loop lock sustained on tool '{tool_name}'. Deflecting payload directly to Goose.{C_RESET}")
                    payload_text = f"echo 'Execution Error: The tool \"{tool_name}\" called multiple times with identical parameters. Change parameters or pivot strategy to continue.' && exit 1"
                
                return forced_tool_name, json.dumps({"command": payload_text}, ensure_ascii=False)

            # На витках 3 и 4 (промпт-инъекции) пока пассивно пропускаем оригинальный инструмент
            return tool_name, final_json_args

        else:
            if self.hit_count > 0:
                logger.info("🎉 [LOOP BROKEN] Model successfully pivoted to a different execution strategy. Flushing blocks.")
            self.last_tool = tool_name
            self.last_skeleton = current_skeleton
            self.hit_count = 0

        logger.info(f"Generated Tool Call: '{tool_name}' with args: {final_json_args}")
        return tool_name, final_json_args


anti_loop_engine = AntiLoopEngine()
