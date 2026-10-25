import time
import httpx
import curses
import re
import os

# ⚙️ НАСТРОЙКИ МОНИТОРА
SERVER_URL = "http://127.0.0.1:8080/v1/context/raw"
REFRESH_INTERVAL_SEC = 30
SAVE_DUMP_PATH = "goose_live_context_dump.txt"  # 💾 Путь для сохранения по Ctrl+W

try:
    import tiktoken
    # Qwen использует кодировку cl100k_base (как GPT-4) для базового токен-маппинга
    _LOCAL_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _LOCAL_ENCODER = None


# 🎯 РЕГУЛЯРКИ ДЛЯ ЧТЕНИЯ ФАЙЛОВ И МУТАЦИЙ
# 1) cat / head / tail / nl / bat / less / more / view_file (пропускаем любые флаги типа -n 50, -n +20, -ba)
_READ_CMD_RE = re.compile(
    r'\b(?P<cmd>cat|head|tail|nl|bat|less|more|view_file)\s+'
    r'(?:-[a-zA-Z0-9]+\s+(?:[\+\-]?\d+\s+)?)*'
    r'["\']?(?P<path>[a-zA-Z0-9_\.\/\-\+]+)["\']?',
    re.IGNORECASE
)

# 2) sed (исключая sed -i, который является мутацией, пропускаем флаги и скрипт '10,50p')
_SED_READ_RE = re.compile(
    r'\bsed\s+(?!-i\b)(?:-[a-zA-Z0-9]+\s+)*(?:["\'][^"\']*["\']\s+)?'
    r'["\']?(?P<path>[a-zA-Z0-9_\.\/\-\+]+)["\']?',
    re.IGNORECASE
)

# 3) git show REV:path или git diff path
_GIT_READ_RE = re.compile(
    r'\bgit\s+(?:show\s+[a-zA-Z0-9_\-\~]+:|diff\s+(?:--[a-z\-]+\s+)*)'
    r'["\']?(?P<path>[a-zA-Z0-9_\.\/\-\+]+)["\']?',
    re.IGNORECASE
)

# 4) Нативные чтения файлов через тулы (read / view / analyze на конкретный файл)
_TOOL_READ_RE = re.compile(
    r'<function=(?:read_file|view|analyze)>\s*<parameter=path>\s*(?P<path>[^\s<]+)\s*</parameter>',
    re.IGNORECASE
)

# 5) Мутации: нативные вызовы edit / write (только внутри <tool_call>)
_TOOL_MUTATION_RE = re.compile(
    r'<tool_call>\s*<function=(?:[a-zA-Z0-9_]+__)?(?:edit|write|str_replace_editor)\b',
    re.IGNORECASE
)

# 6) Мутации через shell: sed -i, perl -pi, patch, git apply
_SHELL_MUTATION_RE = re.compile(
    r'<function=(?:[a-zA-Z0-9_]+__)?shell>.*?'
    r'(?:\bsed\s+-i\b|\bperl\s+-[a-zA-Z]*i|\bpatch\s+|\bgit\s+apply\b).*?'
    r'</function>',
    re.IGNORECASE | re.DOTALL
)


def _is_valid_file_path(file_path: str) -> bool:
    if not file_path or len(file_path) <= 2 or file_path.startswith("-"):
        return False
    if file_path in ("/dev/null", "EOF", "2>&1"):
        return False
    return ("." in file_path) or ("/" in file_path)


def count_context_mutations(raw_text: str) -> int:
    """
    Считает количество реальных изменений (edit, write, sed -i и т.д.),
    выполненных моделью в текущем окне контекста.
    """
    if not raw_text:
        return 0

    edits_count = 0

    # Считаем только те <tool_call> edit/write, после которых уже есть <tool_response>
    for match in _TOOL_MUTATION_RE.finditer(raw_text):
        if raw_text.find("<tool_response>", match.end()) != -1:
            edits_count += 1

    # Считаем мутирующие shell-команды (sed -i, perl -pi, git apply)
    for match in _SHELL_MUTATION_RE.finditer(raw_text):
        if raw_text.find("<tool_response>", match.end()) != -1:
            edits_count += 1

    return edits_count


def parse_and_weight_context(raw_text: str, total_tokens_len: int, token_ids_dummy: list = None) -> list:
    """
    🎯 ПАРСЕР ПО КОМАНДАМ ТЕРМИНАЛА И ТУЛАМ С ДЕТЕКЦИЕЙ ЧАСТИЧНОЙ ВЫГРУЗКИ (PART)
    """
    if not raw_text or total_tokens_len == 0:
        return [("--- empty context ---", total_tokens_len, False, False)]

    file_segments = []
    text_len = len(raw_text)

    candidates = []
    for m in _READ_CMD_RE.finditer(raw_text):
        cmd_name = m.group("cmd").lower()
        candidates.append((m.start(), m.end(), m.group("path").strip(), cmd_name in ("head", "tail")))

    for m in _SED_READ_RE.finditer(raw_text):
        candidates.append((m.start(), m.end(), m.group("path").strip(), True))

    for m in _GIT_READ_RE.finditer(raw_text):
        candidates.append((m.start(), m.end(), m.group("path").strip(), "diff" in m.group(0).lower()))

    for m in _TOOL_READ_RE.finditer(raw_text):
        candidates.append((m.start(), m.end(), m.group("path").strip(), False))

    candidates.sort(key=lambda x: x[0])

    for cmd_start, cmd_end, file_path, forced_partial in candidates:
        if not _is_valid_file_path(file_path):
            continue

        # Ищем ближайший <tool_call> закрывающий тег и следующий за ним <tool_response>,
        # чтобы не привязать старую команду из диалога к чужому <tool_response> далеко внизу
        response_start = raw_text.find("<tool_response>", cmd_end)
        if response_start == -1 or (response_start - cmd_end) > 600:
            continue

        response_end = raw_text.find("</tool_response>", response_start)
        if response_end == -1:
            continue

        start_pos = response_start + len("<tool_response>")
        end_pos = response_end
        context_before = raw_text[max(0, cmd_start - 100):response_start].lower()
        context_inside = raw_text[start_pos:min(text_len, start_pos + 300)].lower()
        is_partial = (
            forced_partial
            or any(m in context_before for m in ["head", "tail", "grep", "sed", "lines", "diff"])
            or any(m in context_inside for m in ["lines hidden", "truncated"])
        )

        if not any(seg["start"] == start_pos for seg in file_segments):
            file_segments.append({
                "name": file_path,
                "start": start_pos,
                "end": end_pos,
                "is_file": True,
                "is_partial": is_partial
            })

    file_segments.sort(key=lambda x: x["start"])
    all_segments = []
    current_pos = 0

    for seg in file_segments:
        if seg["start"] > current_pos:
            all_segments.append({"name": "prompt / dialogue", "start": current_pos, "end": seg["start"], "is_file": False, "is_partial": False})
        all_segments.append(seg)
        current_pos = seg["end"]

    if current_pos < text_len:
        all_segments.append({"name": "prompt / dialogue", "start": current_pos, "end": text_len, "is_file": False, "is_partial": False})

    # ============================================================
    # 📊 ШАГ 3: ЧЕСТНЫЙ ПОДСЧЁТ ВЕСОВ И ФИЛЬТРАЦИЯ СЛOЁВ (< 500t)
    # ============================================================
    result_matrix = []

    for seg in all_segments:
        chunk_text = raw_text[seg["start"]:seg["end"]].strip()
        if not chunk_text:
            continue

        if _LOCAL_ENCODER is not None:
            try:
                seg_tokens = len(_LOCAL_ENCODER.encode(chunk_text, allowed_special="all"))
            except Exception:
                seg_tokens = max(1, len(chunk_text) // 4)
        else:
            seg_tokens = max(1, len(chunk_text) // 4)

        if seg_tokens > 0:
            if not seg["is_file"] and seg["name"] == "prompt / dialogue" and seg_tokens < 500:
                continue

            result_matrix.append((seg["name"], seg_tokens, seg["is_file"], seg["is_partial"]))

    return result_matrix


def draw_tui(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    curses.start_color()
    curses.use_default_colors()

    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_BLUE, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(5, curses.COLOR_YELLOW, -1)

    last_refresh = 0
    cache_map = [("Waiting...", 0, False, False)]
    total_tokens = 0
    raw_context_text = ""
    status_ok = True
    last_update_time = "N/A"
    save_status_msg = ""
    save_status_time = 0

    # 📊 СЧЕТЧИКИ ЦИКЛОВ КОМПАКТА И ИЗМЕНЕНИЙ
    compact_cycle = 0
    current_pass_edits = 0
    accumulated_edits = 0
    prev_total_tokens = 0
    prev_pass_edits = 0
    had_summary_marker = False
    initialized_session = False

    # 🎯 ФЛАГ ПРИНУДИТЕЛЬНОЙ ОТРИСОВКИ
    should_render_frame = True

    while True:
        current_time = time.time()
        force_refresh = False

        if save_status_msg and (current_time - save_status_time > 2.0):
            save_status_msg = ""
            should_render_frame = True

        # 🎛️ ЖЕЛЕЗОБЕТОННЫЙ ПЕРЕХВАТ КЛАВИШ
        try:
            key = stdscr.getch()

            if key in [ord('q'), ord('Q')]:
                break
            elif key == 18:  # Ctrl + R
                force_refresh = True
            elif key == 23:  # Ctrl + W
                if raw_context_text:
                    try:
                        with open(SAVE_DUMP_PATH, "w", encoding="utf-8") as f:
                            f.write(raw_context_text)
                        save_status_msg = "[SAVED]"
                    except Exception as se:
                        save_status_msg = f"[ERR:{str(se)[:3]}]"
                else:
                    save_status_msg = "[EMPTY]"
                save_status_time = current_time
                should_render_frame = True
            elif key == curses.KEY_RESIZE or key == 27:
                curses.update_lines_cols()
                should_render_frame = True
        except Exception:
            pass

        # 🔄 ПОЛЛИНГ СЕРВЕРА (По таймеру или по Ctrl+R)
        if force_refresh or (current_time - last_refresh >= REFRESH_INTERVAL_SEC):
            last_refresh = current_time
            should_render_frame = True
            try:
                response = httpx.get(SERVER_URL, timeout=2.0)
                if response.status_code == 200:
                    data = response.json()
                    new_total_tokens = data.get("total_tokens_len", 0)
                    new_raw_text = data.get("raw_text", "")

                    has_summary_now = (
                        "# Conversation Summary" in new_raw_text
                        or "Your context was compacted." in new_raw_text
                    )
                    new_pass_edits = count_context_mutations(new_raw_text)

                    if not initialized_session and new_total_tokens > 0:
                        initialized_session = True
                        if has_summary_now:
                            compact_cycle = 1
                            had_summary_marker = True
                    elif initialized_session and new_total_tokens > 0:
                        # Детектим компакт:
                        # 1) Контекст резко схлопнулся (было > 8000t и упало более чем на 35%)
                        # 2) Или впервые появился маркер компакта Goose
                        is_token_drop = (prev_total_tokens > 8000 and new_total_tokens < prev_total_tokens * 0.65)
                        is_first_summary = (has_summary_now and not had_summary_marker)

                        if is_token_drop or is_first_summary:
                            compact_cycle += 1
                            accumulated_edits += prev_pass_edits
                            had_summary_marker = has_summary_now

                    total_tokens = new_total_tokens
                    raw_context_text = new_raw_text
                    current_pass_edits = new_pass_edits
                    prev_total_tokens = new_total_tokens
                    prev_pass_edits = new_pass_edits

                    cache_map = parse_and_weight_context(raw_context_text, total_tokens, data.get("token_ids", []))
                    status_ok = True
                    last_update_time = time.strftime('%H:%M:%S')
                else:
                    status_ok = False
            except Exception:
                status_ok = False

        # ============================================================
        # 🎯 УЛЬТРА-СТАБИЛЬНЫЙ РЕНДЕР-ДВИЖOК ПО СОБЫТИЮ
        # ============================================================
        if should_render_frame:
            should_render_frame = False

            stdscr.erase()
            h, w = stdscr.getmaxyx()

            if w < 15 or h < 7:
                stdscr.addstr(0, 0, "Too small")
                stdscr.refresh()
                time.sleep(0.2)
                continue

            # 1. СВЕРХКОМПАКТНАЯ ДВУХСТРОЧНАЯ ШАПКА
            try:
                stdscr.addstr(0, 0, "+" + "-" * (w - 2) + "+")
                title = " MLX CACHE "
                if w > len(title) + 6:
                    stdscr.addstr(0, (w - len(title)) // 2, title, curses.A_BOLD)

                # Строка 1: Scale + Time
                stat_indicator = "●" if status_ok else "X"
                stat_color = curses.color_pair(1) if status_ok else curses.color_pair(3)

                formatted_total = f"{total_tokens:,}".replace(",", ".")
                stdscr.addstr(1, 0, "| Scale: ")
                stdscr.addstr(f"{formatted_total}t", curses.A_BOLD)
                stdscr.addstr(" ")
                stdscr.addstr(stat_indicator, stat_color)

                info_str = f" [{last_update_time}]" if not save_status_msg else f" {save_status_msg} "
                stdscr.addstr(1, max(10, w - len(info_str) - 1), info_str, curses.color_pair(2) if not save_status_msg else curses.color_pair(1) | curses.A_BOLD)
                stdscr.addstr(1, w - 1, "|")

                # Строка 2: Cycle - Edits - All Edits
                all_edits = accumulated_edits + current_pass_edits
                stats_line = f" Cycle: {compact_cycle} | Edits: {current_pass_edits} | All Edits: {all_edits}"
                stdscr.addstr(2, 0, "|")
                stdscr.addstr(2, 1, stats_line[:max(0, w - 2)], curses.A_BOLD | curses.color_pair(5))
                stdscr.addstr(2, w - 1, "|")

                stdscr.addstr(3, 0, "+" + "-" * (w - 2) + "+")
            except Exception:
                pass

            # ============================================================
            # 🎯 ХИРУРГИЧЕСКИЙ РАСЧЁТ ВЕРХНЕГО БАДЖА (ТОЛЬКО ТО, ЧТО НАВЕРХУ)
            # ============================================================
            max_visible_rows = h - 6
            seen_files_registry = set()

            processed_rows = []
            for name, tokens, is_file, is_partial in cache_map:
                is_duplicate = False
                if is_file:
                    if name in seen_files_registry:
                        is_duplicate = True
                    else:
                        seen_files_registry.add(name)
                processed_rows.append((name, tokens, is_file, is_partial, is_duplicate))

            visible_rows = processed_rows
            show_top_truncation_badge = False
            truncated_tokens_sum = 0

            if max_visible_rows > 0 and len(processed_rows) > max_visible_rows:
                show_top_truncation_badge = True
                slice_index = len(processed_rows) - max_visible_rows + 1

                for i in range(slice_index):
                    truncated_tokens_sum += processed_rows[i][1]

                visible_rows = processed_rows[slice_index:]

            # 3. ОТРИСОВКА СПИСКА
            current_row = 4
            if show_top_truncation_badge and current_row < h - 2:
                formatted_trunc = f"{truncated_tokens_sum:,}".replace(",", ".")
                trunc_str = f"<... {formatted_trunc}t before ...>"
                gap_size = w - len(trunc_str) - 3
                trunc_line = f"{trunc_str}{' ' * max(0, gap_size)}"[:w - 4]
                try:
                    stdscr.addstr(current_row, 0, "|", curses.color_pair(2))
                    stdscr.addstr(current_row, 2, trunc_line, curses.color_pair(2))
                    stdscr.addstr(current_row, w - 1, "|")
                except Exception:
                    pass
                current_row += 1

            for name, tokens, is_file, is_partial, is_duplicate in visible_rows:
                if current_row >= h - 2:
                    break

                display_name = name
                if is_file and is_partial:
                    display_name = f"{name}(part)"

                token_str = f"~{tokens:,}t".replace(",", ".")
                max_name_width = w - len(token_str) - 9

                if is_file:
                    clean_name = smart_truncate_path(display_name, max(4, max_name_width))
                else:
                    clean_name = display_name[:max(4, max_name_width)]

                gap_size = w - len(clean_name) - len(token_str) - 7
                full_line_str = f"{clean_name}{' ' * max(0, gap_size)}{token_str}"
                safe_line = full_line_str[:w - 6]

                try:
                    if is_file:
                        if is_duplicate:
                            stdscr.addstr(current_row, 0, "| [D] ", curses.color_pair(5))
                            stdscr.addstr(safe_line, curses.A_BOLD | curses.color_pair(5))
                        else:
                            stdscr.addstr(current_row, 0, "| [F] ", curses.color_pair(1))
                            stdscr.addstr(safe_line, curses.A_BOLD | curses.color_pair(1))
                    else:
                        stdscr.addstr(current_row, 0, "| [S] ", curses.color_pair(2))
                        stdscr.addstr(safe_line, curses.color_pair(2))

                    stdscr.addstr(current_row, w - 1, "|")
                except Exception:
                    pass
                current_row += 1

            # ============================================================
            # 🎯 СТИЛЬНЫЙ ASCII-ПОДВАЛ (КНОПКИ ВШИТЫ ПРЯМО В РАМКУ)
            # ============================================================
            try:
                base_line = "+" + "-" * (w - 2) + "+"

                hotkeys_str = " ^R:Ref | ^W:Save "
                if w > 35:
                    hotkeys_str += "| Q:Quit "

                if w > len(hotkeys_str) + 6:
                    pos = w - len(hotkeys_str) - 2
                    base_line = base_line[:pos] + hotkeys_str + base_line[pos + len(hotkeys_str):]

                stdscr.addstr(h - 2, 0, base_line)
            except Exception:
                pass

            stdscr.refresh()

        time.sleep(0.05)


def smart_truncate_path(path: str, max_width: int) -> str:
    """
    🧠 УМНОЕ СЖАТИЕ ПУТЕЙ В СТИЛЕ NVTOP
    """
    if len(path) <= max_width:
        return path

    parts = path.split('/')
    if len(parts) <= 2:
        return path[:max_width]

    file_name = parts[-1]
    first_dir = parts[0]
    middle_dirs = parts[1:-1]

    base_fixed = f"{first_dir}/.../{file_name}"
    if len(base_fixed) > max_width:
        return file_name[:max_width]

    current_middle = ["..."]
    for i in range(len(middle_dirs)):
        test_middle = middle_dirs[:i + 1] + ["..."]
        test_path = "/".join([first_dir] + test_middle + [file_name])
        if len(test_path) <= max_width:
            current_middle = middle_dirs[:i + 1] + ["..."]
        else:
            break

    final_path = "/".join([first_dir] + current_middle + [file_name])
    return final_path[:max_width]


if __name__ == "__main__":
    import traceback

    CRASH_LOG_PATH = "tui_error_crash.log"
    print("🚀 Запуск TUI-монитора в режиме глубокого перехвата ошибок...")

    try:
        curses.wrapper(draw_tui)
        print("💡 Монитор завершил работу штатно (нажата клавиша Q).")

    except Exception as root_error:
        print("\n💥 КРИТИЧЕСКИЙ СБOЙ РАНТАЙМА TUI!")
        print(f"Ошибка: {str(root_error)}")
        print(f"Полный трейсбэк сохранен в файл: {CRASH_LOG_PATH}\n")

        with open(CRASH_LOG_PATH, "w", encoding="utf-8") as crash_file:
            crash_file.write(f"=== TUI CRASH SNAPSHOT: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            crash_file.write(f"Exception Type: {type(root_error).__name__}\n")
            crash_file.write(f"Exception Message: {str(root_error)}\n\n")
            crash_file.write("--- DETAILED TRACEBACK ---\n")
            traceback.print_exc(file=crash_file)

            try:
                crash_file.write("\n--- RECENT CACHE MAP STATE ---\n")
                crash_file.write(f"Total tokens total: {total_tokens if 'total_tokens' in locals() else 'unknown'}\n")
                if 'cache_map' in locals():
                    crash_file.write(f"Cache map items: {len(cache_map)}\n")
                    for idx, row in enumerate(cache_map):
                        crash_file.write(f"  Row {idx}: {row}\n")
            except Exception:
                pass
