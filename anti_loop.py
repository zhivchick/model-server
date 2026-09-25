import re
import json
import logging

logger = logging.getLogger("mlx_lm_server.antiloop")

C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_RED    = "\033[91m"
C_YELLOW = "\033[93m"
C_GREEN  = "\033[92m"
C_CYAN   = "\033[96m"

class AntiLoopEngine:
    """Manages consecutive repetition blocks and triggers smart assists, user interventions, or hard brakes."""
    def __init__(self):
        self.last_tool = None
        self.last_skeleton = None
        self.hit_count = 0

    def _protect_github_entity_ids(self, text: str) -> str:
        """
        Protects specific GitHub entity IDs (issue, PR, run, workflow numbers)
        from being stripped as loop digits, while preserving loop detection
        for pagination flags (-L 10 / --limit 20) and sliding window reads (head/sed).
        """
        def _encode_id(match):
            prefix = match.group(1)
            num = match.group(2)
            # Map digits to letters so re.sub(r'\d+', '', ...) won't erase them
            encoded = "GHID" + num.translate(str.maketrans("0123456789", "abcdefghij"))
            return f"{prefix}{encoded}"

        # gh issue/pr/run/workflow/release/discussion <subcommand> [#]<id>
        # e.g.: gh issue view 24, gh pr diff 25, gh run view #123, gh issue comment 24
        pattern1 = r'\b(gh\s+(?:issue|pr|run|workflow|release|discussion)\s+(?:view|diff|comment|edit|close|reopen|checkout|ready|review|status|rerun|watch|cancel)\s+[^\d\n;&|]*?#?)(\d+)\b'
        text = re.sub(pattern1, _encode_id, text)

        # gh api .../(issues|pulls|runs)/<id>
        pattern2 = r'(\bgh\s+api\s+[^\s;&|]*/(?:issues|pulls|runs)/)(\d+)\b'
        text = re.sub(pattern2, _encode_id, text)

        # Shorthand: gh issue 24, gh pr 24
        pattern3 = r'\b(gh\s+(?:issue|pr)\s+#?)(\d+)\b'
        text = re.sub(pattern3, _encode_id, text)

        # Standalone issue/PR refs like issue #24, PR #25
        pattern4 = r'\b((?:issue|pr|run)\s*#)(\d+)\b'
        text = re.sub(pattern4, _encode_id, text, flags=re.IGNORECASE)

        # Structured tool arguments for GitHub integrations
        pattern5 = r'("(?:issue|issue_id|issue_number|pr|pr_id|pr_number|pull_request)":\s*)(\d+)\b'
        text = re.sub(pattern5, _encode_id, text)

        return text

    def _build_argument_skeleton(self, args_dict: dict) -> str:
        raw_str = json.dumps(args_dict, ensure_ascii=False)
        protected_str = self._protect_github_entity_ids(raw_str)
        return re.sub(r'\d+', '', protected_str)

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
        # СИММЕТРИЧНАЯ МАТРИЦА ПOВТOРOВ (3 ЭШЕЛОНА ЗАЩИТЫ)
        # ------------------------------------------------------------
        if self.last_tool == tool_name and self.last_skeleton == current_skeleton:
            self.hit_count += 1
            logger.warning(f"{C_YELLOW}⚠️ [ANTI-LOOP FIREWALL] Repetitive pattern found! Tool: '{tool_name}', Depth: {self.hit_count}{C_RESET}")
            
            # 🚨 ЭШЕЛОН 3: КРИТИЧЕСКИЙ СТОП-КРАН (Попытка 6, hit_count >= 5) — ОКОНЧАТЕЛЬНЫЙ ОБРЫВ
            if self.hit_count >= 5:
                logger.error(f"{C_BOLD}{C_RED}🚨 [CONTEXT EMERGENCY] Hard threshold reached on tool '{tool_name}' (Depth {self.hit_count}). Forcing Dialogue Brake!{C_RESET}")
                self.hit_count = 0
                self.last_tool = None
                self.last_skeleton = None
                
                compact_trigger_text = (
                    f"⚠️ [SERVER NOTICE] Critical repetition loop detected on tool '{tool_name}'. "
                    "Forcing thread synchronization break to return control to the human user and trigger active memory compacting routines."
                )
                return None, compact_trigger_text

            forced_tool_name = "shell"

            # 👤 ЭШЕЛОН 2: ВТОРЫЕ ДВА ДУБЛЯ (Вызовы 4 и 5, hit_count == 3 и 4) — ОТВЕТ ОТ ИМЕНИ ЮЗЕРА
            if self.hit_count == 3:
                logger.error(f"{C_BOLD}{C_CYAN}👤 [USER INTERVENTION - CALL 4] Tool '{tool_name}' repeated 3 times. Injecting human operator intervention.{C_RESET}")
                payload_text = (
                    f"[USER INTERVENTION]: Stop! You have called the tool \"{tool_name}\" 3 times in a row with identical parameters without making progress. "
                    f"I am intervening directly as the user: do NOT retry this exact command. "
                    f"Analyze the previous outputs, change your strategy, or ask me for clarification."
                )
            elif self.hit_count == 4:
                logger.error(f"{C_BOLD}{C_RED}👤 [USER DIRECTIVE - CALL 5] Final warning before session break on tool '{tool_name}'. Injecting strict user directive.{C_RESET}")
                payload_text = (
                    f"[USER DIRECTIVE - FINAL WARNING]: You are ignoring instructions and still attempting to call \"{tool_name}\". "
                    f"This is your final warning: do NOT invoke \"{tool_name}\" again with these arguments. "
                    f"Summarize what is blocking you or switch to an entirely different approach now, or the session will be terminated."
                )
            # 🛠️ ЭШЕЛОН 1: ПЕРВЫЕ ДВА ДУБЛЯ (Вызовы 2 и 3, hit_count == 1 и 2) — ПОДМЕНА ОТВЕТА ТУЛА
            elif is_idle_edit:
                logger.error(f"{C_YELLOW}🚨 [ANTI-LOOP FIREWALL] Идентичные before/after на витке {self.hit_count}. Отклоняем payload.{C_RESET}")
                payload_text = (
                    "Execution Error: The \"before\" and \"after\" parameters are byte-for-byte identical. "
                    "Your edit action did NOT change any code. Rewrite your \"after\" block to apply real modifications or use another tool."
                )
            elif is_edit_tool:
                logger.error(f"{C_GREEN}💡 [SMART ASSIST] Continuous ambiguous edit detected! Injecting guide.{C_RESET}")
                if self.hit_count == 1:
                    payload_text = (
                        "Execution Error: The block you provided in the \"before\" parameter matches multiple lines in the file. "
                        "Do NOT repeat the exact same \"before\" string. To fix this, look at the Match lines and rewrite your edit call "
                        "by including 2-3 lines of surrounding code ABOVE and BELOW the target line inside both \"before\" and \"after\" parameters to make it unique."
                    )
                else:
                    payload_text = (
                        "Execution Error: Ambiguity loop sustained. Rewrite your edit call by expanding the surrounding context window "
                        "(at least 3-5 unique lines above and below) or inspect the file with a read tool first."
                    )
            else:
                # Стандартный жесткий блок для shell / ls / cat на Depth 1 и 2
                logger.error(f"{C_YELLOW}🚨 [FIREWALL BRICKWALL] Continuous loop lock sustained on tool '{tool_name}'. Deflecting payload.{C_RESET}")
                if self.hit_count == 1:
                    payload_text = (
                        f"Execution Error: The tool \"{tool_name}\" called multiple times with the same parameters. "
                        f"Change parameters or use another tool to continue."
                    )
                else:
                    payload_text = (
                        f"Execution Error: Repeated call signature confirmed for tool \"{tool_name}\". "
                        f"Your previous call produced identical results. Pivot your command or inspect other files to proceed."
                    )
                
            safe_payload = payload_text.replace("'", "\\'")
            shell_cmd = f"echo '{safe_payload}' && exit 1"
            return forced_tool_name, json.dumps({"command": shell_cmd}, ensure_ascii=False)

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
