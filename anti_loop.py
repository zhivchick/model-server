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

    def _detect_echo_reflection(self, tool_name: str, extracted_args_dict: dict, full_text: str) -> bool:
        """
        Проверяет, не пытается ли модель выполнить команду echo, содержащую сигнатуры
        системных сообщений об ошибках нашего собственного файрвола.
        """
        cmd_candidates = []
        if isinstance(extracted_args_dict, dict):
            for k in ("command", "cmd", "raw_arguments"):
                v = extracted_args_dict.get(k)
                if isinstance(v, str):
                    cmd_candidates.append(v)
        
        if not cmd_candidates and full_text:
            cmd_candidates.append(full_text)

        firewall_signatures = [
            "execution error",
            "called multiple times",
            "repeated call signature",
            "user intervention",
            "user directive",
            "continuous loop lock",
            "deflecting payload",
            "pivot your command",
            "change parameters or use another tool",
            "byte-for-byte identical",
            "matches multiple lines",
            "ambiguity loop",
        ]

        for text in cmd_candidates:
            if re.search(r'\becho\b', text, re.IGNORECASE):
                lower_text = text.lower()
                for sig in firewall_signatures:
                    if sig in lower_text:
                        return True

        return False

    def evaluate_and_process(self, full_text: str, parser_tool_name: str, extracted_args_dict: dict) -> tuple:
        # Определяем реальное имя тула
        func_match = re.search(r'<?function=([^>]+)>?', full_text)
        if func_match:
            tool_name = func_match.group(1).strip().replace('"', '').replace("'", "")
        else:
            tool_name = parser_tool_name if parser_tool_name else "shell"

        # ------------------------------------------------------------
        # 🛑 ДЕТЕКТОР ЭХО-ГАЛЛЮЦИНАЦИЙ ОШИБОК ФАЙРВОЛА (ECHO REFLECTION TRAP)
        # Если модель пытается вызвать 'echo' с текстом системной ошибки сервера,
        # блокируем вызов в Goose и мгновенно возвращаем текст от имени юзера.
        # ------------------------------------------------------------
        if self._detect_echo_reflection(tool_name, extracted_args_dict, full_text):
            logger.error(
                f"{C_BOLD}{C_RED}🛑 [ECHO REFLECTION TRAP] Модель пытается выполнить 'echo' с текстом ошибки файрвола! "
                f"Блокируем тул-колл в Goose, возвращаем прямое вмешательство пользователя.{C_RESET}"
            )
            self.hit_count = 0
            self.last_tool = None
            self.last_skeleton = None
            reflection_text = (
                "[USER INTERVENTION]: You are echoing the server's previous execution error back as a shell command! "
                "Immediately STOP calling 'echo' with error messages. "
                "Analyze what actually needs to be done, change your strategy, and use legitimate tools."
            )
            return None, reflection_text

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
        # СИММЕТРИЧНАЯ МАТРИЦА ПOВТOРOВ (3 ЭШЕЛОНА ЗАЩИТЫ: 1/5 .. 5/5)
        # ------------------------------------------------------------
        if self.last_tool == tool_name and self.last_skeleton == current_skeleton:
            self.hit_count += 1
            
            # 🚨 ЭШЕЛОН 3: КРИТИЧЕСКИЙ СТОП-КРАН (Повтор 5/5, hit_count >= 5) — ОКОНЧАТЕЛЬНЫЙ ОБРЫВ
            if self.hit_count >= 5:
                logger.error(f"{C_BOLD}{C_RED}🚨 [CONTEXT EMERGENCY] Повтор 5/5: Лимит повторов исчерпан на туле '{tool_name}'. Аварийный стоп (Dialogue Brake)!{C_RESET}")
                self.hit_count = 0
                self.last_tool = None
                self.last_skeleton = None
                
                compact_trigger_text = (
                    f"⚠️ [SERVER NOTICE] Critical repetition loop detected on tool '{tool_name}' (5/5). "
                    "Forcing thread synchronization break to return control to the human user and trigger active memory compacting routines."
                )
                return None, compact_trigger_text

            forced_tool_name = "shell"

            # 👤 ЭШЕЛОН 2: ВТОРЫЕ ДВА ДУБЛЯ (3/5 и 4/5) — ОТВЕТ ОТ ИМЕНИ ЮЗЕРА
            if self.hit_count == 3:
                logger.error(f"{C_BOLD}{C_CYAN}👤 [USER INTERVENTION - 3/5] Тул '{tool_name}' повторен 3-й раз. Модель проигнорировала подсказку. Отклоняем payload с [USER INTERVENTION].{C_RESET}")
                payload_text = (
                    f"[USER INTERVENTION]: Stop! You have called the tool \"{tool_name}\" 3 times in a row with identical parameters without making progress. "
                    f"I am intervening directly as the user: do NOT retry this exact command. "
                    f"Analyze the previous outputs, change your strategy, or ask me for clarification."
                )
            elif self.hit_count == 4:
                logger.error(f"{C_BOLD}{C_RED}👤 [USER DIRECTIVE - 4/5] Тул '{tool_name}' повторен 4-й раз. Финальное предупреждение перед аварийным стопом 5/5. Отклоняем payload.{C_RESET}")
                payload_text = (
                    f"[USER DIRECTIVE - FINAL WARNING]: You are ignoring instructions and still attempting to call \"{tool_name}\". "
                    f"This is your final warning: do NOT invoke \"{tool_name}\" again with these arguments. "
                    f"Summarize what is blocking you or switch to an entirely different approach now, or the session will be terminated."
                )
            # 🛠️ ЭШЕЛОН 1: ПЕРВЫЕ ДВА ДУБЛЯ (1/5 и 2/5) — ПОДМЕНА ОТВЕТА ТУЛА
            elif is_idle_edit:
                logger.warning(f"{C_YELLOW}⚠️ [ANTI-LOOP FIREWALL] Повтор {self.hit_count}/5: Идентичные before/after у '{tool_name}'. Отклоняем payload с ошибкой.{C_RESET}")
                payload_text = (
                    "Execution Error: The \"before\" and \"after\" parameters are byte-for-byte identical. "
                    "Your edit action did NOT change any code. Rewrite your \"after\" block to apply real modifications or use another tool."
                )
            elif is_edit_tool:
                logger.warning(f"{C_YELLOW}⚠️ [ANTI-LOOP FIREWALL] Повтор {self.hit_count}/5: Зацикливание правки '{tool_name}'. Внедряем подсказку расширить контекст.{C_RESET}")
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
                # Стандартный блок для shell / ls / cat на 1/5 и 2/5
                escalation_hint = " (Внимание: следующий повтор 3/5 подключит оператора [USER INTERVENTION])" if self.hit_count == 2 else ""
                logger.warning(f"{C_YELLOW}⚠️ [ANTI-LOOP FIREWALL] Повтор {self.hit_count}/5: Тул '{tool_name}' вызван повторно. Подменяем ответ тула на Execution Error.{escalation_hint}{C_RESET}")
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
                logger.info(f"{C_GREEN}🎉 [LOOP BROKEN] Модель успешно сменила стратегию (вызов '{tool_name}'). Счетчик повторов ({self.hit_count}/5) сброшен.{C_RESET}")
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
