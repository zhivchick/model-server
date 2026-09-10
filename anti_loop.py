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
    """Manages consecutive repetition blocks and triggers smart assists or loop blocks synchronously."""
    def __init__(self):
        self.last_tool = None
        self.last_skeleton = None
        self.hit_count = 0

    def _build_argument_skeleton(self, args_dict: dict) -> str:
        raw_str = json.dumps(args_dict, ensure_ascii=False)
        return re.sub(r'\d+', '', raw_str)

    def evaluate_and_process(self, full_text: str, parser_tool_name: str, extracted_args_dict: dict) -> tuple:
        # Определяем реальное имя тула
        func_match = re.search(r'<?function=([^>]+)>?', full_text)
        if func_match:
            tool_name = func_match.group(1).strip().replace('"', '').replace("'", "")
        else:
            tool_name = parser_tool_name if parser_tool_name else "shell"

        # Проверяем, является ли инструмент правкой кода
        is_edit_tool = tool_name in ["edit", "developer__edit", "patch"]
        
        # ДЕТЕКТОР ИДЕНТИЧНЫХ ПРАВОК (ХОЛОСТОЙ ВЫЗОВ)
        is_idle_edit = False
        if is_edit_tool:
            before_str = extracted_args_dict.get("before", "")
            after_str = extracted_args_dict.get("after", "")
            if before_str and after_str and before_str.strip() == after_str.strip():
                is_idle_edit = True

        # Строим скелет для матрицы повторов. 
        # Если это холостая правка, мы принудительно помечаем скелет специальным маркером,
        # чтобы даже если аргументы внутри before/after плавают, файрвол видел ХОЛОСТУЮ ПЕТЛЮ.
        if is_idle_edit:
            current_skeleton = "IDLE_EDIT_DETECTED_LOOP_MARKER"
        else:
            current_skeleton = self._build_argument_skeleton(extracted_args_dict)

        final_json_args = json.dumps(extracted_args_dict, ensure_ascii=False)

        # ------------------------------------------------------------
        # СИММЕТРИЧНАЯ МАТРИЦА ПOВТOРOВ (ТЕПЕРЬ ХОЛОСТЫЕ ПРАВКИ КОПЯТ ХИТЫ!)
        # ------------------------------------------------------------
        if self.last_tool == tool_name and self.last_skeleton == current_skeleton:
            self.hit_count += 1
            logger.warning(f"{C_YELLOW}⚠️ [ANTI-LOOP FIREWALL] Repetitive pattern found! Tool: '{tool_name}', Depth: {self.hit_count}{C_RESET}")
            
            # 🚨 КРИТИЧЕСКИЙ СТOП-КРАН НА ГЛУБИНЕ 3 (Четвертый повтор) - ЕДИНЫЙ ДЛЯ ВСЕХ
            if self.hit_count >= 3:
                logger.error(f"{C_BOLD}{C_RED}🚨 [CONTEXT EMERGENCY] Hard threshold reached on tool '{tool_name}'. Forcing Dialogue Brake!{C_RESET}")
                self.hit_count = 0
                self.last_tool = None
                self.last_skeleton = None
                
                compact_trigger_text = (
                    "⚠️ [SERVER NOTICE] Attention window degradation detected due to context scale. "
                    "Forcing thread synchronization break to trigger active memory compacting routines."
                )
                return None, compact_trigger_text

            # 🛠️ ДИНАМИЧЕСКИЕ ШАГИ НА ВИТКАХ 1 И 2 (Вброс ответов в модель)
            forced_tool_name = "shell"
            
            if is_idle_edit:
                logger.error(f"{C_YELLOW}🚨 [ANTI-LOOP FIREWALL] Идентичные before/after на витке {self.hit_count}. Отклоняем payload.{C_RESET}")
                payload_text = (
                    "Execution Error: The \"before\" and \"after\" parameters are byte-for-byte identical. "
                    "Your edit action did NOT change any code. Rewrite your \"after\" block to apply real modifications or use another tool."
                )
            elif is_edit_tool:
                logger.error(f"{C_GREEN}💡 [SMART ASSIST] Continuous ambiguous edit detected! Injecting guide.{C_RESET}")
                payload_text = (
                    "The block you provided in the \"before\" parameter matches multiple lines in the file."
                    "Do NOT repeat the exact same \"before\" string. To fix this, look at the Match lines and rewrite your edit call"
                    "by including 2-3 lines of surrounding code ABOVE and BELOW the target line inside both \"before\" and \"after\" parameters to make it unique."
                )
            else:
                # Стандартный жесткий блок для shell / ls / cat
                logger.error(f"{C_YELLOW}🚨 [FIREWALL BRICKWALL] Continuous loop lock sustained on tool '{tool_name}'. Deflecting payload.{C_RESET}")
                payload_text = (
                    f"echo 'Execution Error: The tool \"{tool_name}\" called multiple times with the same parameters. "
                    f"Change parameters or use another tool to continue' && exit 1"
                )
                
            return forced_tool_name, json.dumps({"command": f"echo '{payload_text}' && exit 1" if not forced_tool_name == "shell" else payload_text}, ensure_ascii=False)

        else:
            # Свежий шаг — обновляем стейт блокировок
            if self.hit_count > 0:
                logger.info("🎉 [LOOP BROKEN] Model successfully pivoted to a different execution strategy. Flushing firewall blocks.")
            self.last_tool = tool_name
            self.last_skeleton = current_skeleton
            self.hit_count = 0

        # Если это холостой вызов на самом первом витке (hit_count == 0)
        if is_idle_edit:
            logger.error(f"{C_YELLOW}🚨 [EDIT IDLE DETECTED] Модель прислала идентичные блоки на витке 0. Отклоняем.{C_RESET}")
            forced_tool_name = "shell"
            idle_warning_text = (
                "echo 'Execution Error: The \"before\" and \"after\" parameters are byte-for-byte identical. "
                "Your edit action did NOT change any code. Rewrite your \"after\" block to apply real modifications or use another tool.' && exit 1"
                )
            return forced_tool_name, json.dumps({"command": idle_warning_text}, ensure_ascii=False)

        logger.info(f"Generated Tool Call: '{tool_name}' with args: {final_json_args}")
        return tool_name, final_json_args

anti_loop_engine = AntiLoopEngine()
