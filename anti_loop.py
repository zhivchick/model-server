import re
import json
import logging

logger = logging.getLogger("mlx_lm_server.antiloop")

C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_RED    = "\033[91m"
C_YELLOW = "\033[93m"
C_GREEN  = "\033[92m"

class AntiLoopEngine:
    def __init__(self):
        self.last_tool = None
        self.last_skeleton = None
        self.hit_count = 0

    def _build_argument_skeleton(self, args_dict: dict) -> str:
        raw_str = json.dumps(args_dict, ensure_ascii=False)
        return re.sub(r'\d+', '', raw_str)

    def evaluate_and_process(self, full_text: str, parser_tool_name: str, extracted_args_dict: dict) -> tuple:
        func_match = re.search(r'<?function=([^>]+)>?', full_text)
        if func_match:
            tool_name = func_match.group(1).strip().replace('"', '').replace("'", "")
        else:
            tool_name = parser_tool_name if parser_tool_name else "shell"

        current_skeleton = self._build_argument_skeleton(extracted_args_dict)
        final_json_args = json.dumps(extracted_args_dict, ensure_ascii=False)

        # Проверяем, не застряли ли мы конкретно на инструменте правки кода
        is_edit_tool = tool_name in ["edit", "developer__edit", "patch"]

        if self.last_tool == tool_name and self.last_skeleton == current_skeleton:
            self.hit_count += 1
            logger.warning(f"{C_YELLOW}⚠️ [ANTI-LOOP FIREWALL] Repetitive pattern found! Tool: '{tool_name}', Depth: {self.hit_count}{C_RESET}")
            
            # 🎯 ИНТЕЛЛЕГЕНТНЫЙ ПЕРЕХВАТ ДЛЯ EDIT (Attempt 2+)
            # Если модель тупит и присылает ТУ ЖЕ САМУЮ неуникальную правку,
            # мы подменяем вызов на shell с жесткой инструкцией работы над ошибками!
            if is_edit_tool and self.hit_count >= 2:
                logger.error(f"{C_BOLD}{C_GREEN}💡 [SMART ASSIST] Ambiguous edit loop detected! Injecting prompt correction strategy.{C_RESET}")
                
                # Формируем команду, которая вернет модели в чат понятное руководство к действию
                payload_instruction = (
                    "echo 'Execution Error: The block you provided in the \"before\" parameter matches multiple lines in the file. "
                    "Do NOT repeat the exact same \"before\" string. "
                    "To fix this, look at the Match lines and rewrite your edit call by including 2-3 lines of surrounding code "
                    "ABOVE and BELOW the target line inside both \"before\" and \"after\" parameters to make it unique.' && exit 1"
                )
                
                # Сбрасываем счетчик, давая модели шанс исправиться на следующем шагу с новыми знаниями
                self.hit_count = 0
                return "shell", json.dumps({"command": payload_instruction}, ensure_ascii=False)

            # EMERGENCY DIALOGUE BRAKE (для остальных затупов вроде shell на 4-й попытке)
            if self.hit_count >= 4:
                logger.error(f"{C_BOLD}{C_RED}🚨 [CONTEXT EMERGENCY] Hard threshold reached. Forcing Dialogue Brake!{C_RESET}")
                self.hit_count = 0
                self.last_tool = None
                self.last_skeleton = None
                
                compact_trigger_text = (
                    "⚠️ [SERVER NOTICE] Attention window degradation detected due to context scale. "
                    "Forcing thread synchronization break to trigger active memory compacting routines."
                )
                return None, compact_trigger_text

            # Стандартная блокировка для бесконечных shell/ls команд на 2-й и 3-й попытках
            if self.hit_count >= 2:
                logger.error(f"{C_BOLD}{C_RED}🚨 [FIREWALL BRICKWALL] Continuous loop lock sustained on tool '{tool_name}'. Deflecting payload.{C_RESET}")
                forced_tool_name = "shell"
                payload_warning_text = (
                    f"echo 'Execution Error: The tool \"{tool_name}\" is locked in an infinite repetition loop. "
                    f"Action blocked by firewalled server framework constraints. Change your arguments"
                    f"or tool usage strategy completely to proceed.' && exit 1"
                )
                return forced_tool_name, json.dumps({"command": payload_warning_text}, ensure_ascii=False)
        else:
            if self.hit_count > 0:
                logger.info("🎉 [LOOP BROKEN] Model successfully pivoted to a different execution strategy. Flushing firewall blocks.")
            self.last_tool = tool_name
            self.last_skeleton = current_skeleton
            self.hit_count = 0

        logger.info(f"Generated Tool Call: '{tool_name}' with args: {final_json_args}")
        return tool_name, final_json_args

anti_loop_engine = AntiLoopEngine()
