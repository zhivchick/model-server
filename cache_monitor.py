import time
import httpx
import curses
import re
import os

# ⚙️ НАСТРОЙКИ МОНИТОРА
SERVER_URL = "http://127.0.0.1:8080/v1/context/raw"
REFRESH_INTERVAL_SEC = 10
SAVE_DUMP_PATH = "goose_live_context_dump.txt"  # 💾 Путь для сохранения по Ctrl+S
# Добавьте импорт на самый верх cache_monitor.py
try:
    import tiktoken
    # Qwen использует кодировку cl100k_base (как GPT-4) для базового токен-маппинга
    _LOCAL_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _LOCAL_ENCODER = None

def parse_and_weight_context(raw_text: str, total_tokens_len: int, token_ids_dummy: list = None) -> list:
    """
    🎯 ПАРСЕР ПО КОМАНДАМ ТЕРМИНАЛА С ДЕТЕКЦИЕЙ ЧАСТИЧНОЙ ВЫГРУЗКИ (PART)
    """
    if not raw_text or total_tokens_len == 0:
        return [("--- empty context ---", total_tokens_len, False, False)]

    file_segments = []
    text_len = len(raw_text)
    cmd_pattern = r'\bHeader|\b(?:cat|view_file)\s+([a-zA-Z0-9_\.\/\-\+]+)'
    
    for cmd_match in re.finditer(cmd_pattern, raw_text, re.IGNORECASE):
        if len(cmd_match.groups()) == 0 or cmd_match.group(1) is None:
            continue
        file_path = cmd_match.group(1).strip()
        if len(file_path) <= 3 or not ('.' in file_path or '/' in file_path):
            continue
            
        search_start = cmd_match.end()
        response_start = raw_text.find("<tool_response>", search_start)
        if response_start != -1:
            response_end = raw_text.find("</tool_response>", response_start)
            if response_end != -1:
                start_pos = response_start + len("<tool_response>")
                end_pos = response_end
                context_before = raw_text[max(0, cmd_match.start()-100):response_start].lower()
                context_inside = raw_text[start_pos:min(text_len, start_pos+300)].lower()
                is_partial = any(m in context_before for m in ["head", "tail", "grep", "sed", "lines"]) or \
                             any(m in context_inside for m in ["lines hidden", "truncated"])
                
                if not any(seg["start"] == start_pos for seg in file_segments):
                    file_segments.append({"name": file_path, "start": start_pos, "end": end_pos, "is_file": True, "is_partial": is_partial})

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

     # 🎯 ФЛАГ ПРИНУДИТЕЛЬНОЙ ОТРИСОВКИ: Взводим True на старте, чтобы нарисовать первый кадр
    should_render_frame = True

    while True:
        current_time = time.time()
        force_refresh = False
        
        if save_status_msg and (current_time - save_status_time > 2.0):
            save_status_msg = ""
            should_render_frame = True

        # 🎛️ ЖЕЛЕЗОБЕТОННЫЙ ПЕРЕХВАТ КЛАВИШ (ЗАЩИТА ОТ МГНОВЕННОГО ВЫЛЕТА)
        try:
            key = stdscr.getch()
            
            # Строго проверяем конкретные коды кнопок. 
            # Если getch() выдал -1 (нет нажатия) или мусор — игнорируем, цикл НЕ прерывается.
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
            elif key == curses.KEY_RESIZE or key == 27:  # Нативный ресайз терминала Mac
                # Принудительно заставляем curses обновить геометрию экрана
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
                    total_tokens = data.get("total_tokens_len", 0)
                    raw_context_text = data.get("raw_text", "")
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
            should_render_frame = False # Сбрасываем флаг до следующего события
            
            stdscr.erase()
            h, w = stdscr.getmaxyx()

            if w < 15 or h < 6:
                stdscr.addstr(0, 0, "Too small")
                stdscr.refresh()
                time.sleep(0.2)
                continue

            # 1. СВЕРХКОМПАКТНАЯ ШАПКА
            try:
                stdscr.addstr(0, 0, "+" + "-"*(w-2) + "+")
                title = " MLX CACHE "
                if w > len(title) + 6:
                    stdscr.addstr(0, (w-len(title))//2, title, curses.A_BOLD)
                
                stat_indicator = "●" if status_ok else "X"
                stat_color = curses.color_pair(1) if status_ok else curses.color_pair(3)
                
                formatted_total = f"{total_tokens:,}".replace(",", ".")
                stdscr.addstr(1, 1, "| Scale: ")
                stdscr.addstr(f"{formatted_total}t", curses.A_BOLD)
                stdscr.addstr(" ")
                stdscr.addstr(stat_indicator, stat_color)
                
                info_str = f" [{last_update_time}]" if not save_status_msg else f" {save_status_msg} "
                stdscr.addstr(1, max(10, w - len(info_str) - 1), info_str, curses.color_pair(2) if not save_status_msg else curses.color_pair(1) | curses.A_BOLD)
                stdscr.addstr(1, w-1, "|")
                stdscr.addstr(2, 0, "+" + "-"*(w-2) + "+")
            except Exception: pass

            # 2. ВЫЧИСЛЕНИЕ ВЕРТИКАЛЬНОЙ ВЫРЕЗКИ И СЕРВЕРНОГО ТРЕКЕРА ПОВТОРОВ
                        # ============================================================
            # 🎯 ИСПРАВЛЕННЫЙ РАСЧЁТ ВЕРТИКАЛЬНОЙ РЕЗКИ (БЕЗ ИСЧЕЗНОВЕНИЯ СТРОК)
            # ============================================================
            max_visible_rows = h - 5
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

            truncated_tokens_sum = 0
            show_top_truncation_badge = False
            
            # 🎯 ФИКС: По умолчанию видимые строки ВСЕГДА равны обработанным!
            visible_rows = processed_rows

            # Режем историю НАЧАЛА только тогда, когда она РЕАЛЬНО не влезает в экран
            if max_visible_rows > 0 and len(processed_rows) > max_visible_rows:
                show_top_truncation_badge = True
                # Оставляем одну строку под синий бадж усечения
                slice_index = len(processed_rows) - max_visible_rows + 1
                
                # Суммируем токены улетающих вверх строк (берём строго индекс 1)
                for i in range(slice_index):
                    truncated_tokens_sum += processed_rows[i][1]
                    
                visible_rows = processed_rows[slice_index:]
            # ============================================================


            # 3. ОТРИСОВКА СПИСКА
            current_row = 3
            if show_top_truncation_badge and current_row < h - 2:
                formatted_trunc = f"{truncated_tokens_sum:,}".replace(",", ".")
                trunc_str = f"<... {formatted_trunc}t before ...>"
                gap_size = w - len(trunc_str) - 3
                trunc_line = f"{trunc_str}{' ' * max(0, gap_size)}"[:w-4]
                try:
                    stdscr.addstr(current_row, 1, "|", curses.color_pair(2))
                    stdscr.addstr(current_row, 3, trunc_line, curses.color_pair(2))
                    stdscr.addstr(current_row, w-1, "|")
                except Exception: pass
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
                safe_line = full_line_str[:w-6]

                try:
                    if is_file:
                        if is_duplicate:
                            stdscr.addstr(current_row, 1, "| [D] ", curses.color_pair(5))
                            stdscr.addstr(safe_line, curses.A_BOLD | curses.color_pair(5))
                        else:
                            stdscr.addstr(current_row, 1, "| [F] ", curses.color_pair(1))
                            stdscr.addstr(safe_line, curses.A_BOLD | curses.color_pair(1))
                    else:
                        stdscr.addstr(current_row, 1, "| [S] ", curses.color_pair(2))
                        stdscr.addstr(safe_line, curses.color_pair(2))
                    
                    stdscr.addstr(current_row, w-1, "|")
                except Exception: pass
                current_row += 1

            # 4. НИЖНЯЯ РАМКА И ПОДВАЛ APTOP
            try:
                stdscr.addstr(h-2, 0, "+" + "-"*(w-2) + "+")
            except Exception: pass

            try:
                stdscr.addstr(h-1, 0, " " * w, curses.color_pair(4))
                panel_str = " ^R Ref"
                if w > 18:
                    panel_str += "   ^W Save"
                if w > 30:
                    panel_str += f"{' ' * (w - len(panel_str) - 7)}^Q Quit"
                safe_panel = panel_str[:w]
                stdscr.insstr(h-1, 0, safe_panel, curses.color_pair(4))
            except Exception: pass

            stdscr.refresh()

        # Маленький базовый цикл поллинга клавиш
        time.sleep(0.05)



def smart_truncate_path(path: str, max_width: int) -> str:
    """
    🧠 УМНОЕ СЖАТИЕ ПУТЕЙ В СТИЛЕ NVTOP
    Приоритет 1: Имя файла (всегда держим до конца).
    Приоритет 2: Первая папка.
    Приоритет 3: Вторая, третья папки. Всё, что не влезает -> '...'
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
        test_middle = middle_dirs[:i+1] + ["..."]
        test_path = "/".join([first_dir] + test_middle + [file_name])
        if len(test_path) <= max_width:
            current_middle = middle_dirs[:i+1] + ["..."]
        else:
            break

    final_path = "/".join([first_dir] + current_middle + [file_name])
    return final_path[:max_width]

if __name__ == "__main__":
    import traceback
    
    CRASH_LOG_PATH = "tui_error_crash.log"
    print("🚀 Запуск TUI-монитора в режиме глубокого перехвата ошибок...")
    
    try:
        # Пытаемся запустить стандартный curses цикл
        curses.wrapper(draw_tui)
        print("💡 Монитор завершил работу штатно (нажата клавиша Q).")
        
    except Exception as root_error:
        # Если внутри curses что-то ломается — перехватываем, 
        # принудительно восстанавливаем терминал и пишем дамп на диск
        print("\n💥 КРИТИЧЕСКИЙ СБOЙ РАНТАЙМА TUI!")
        print(f"Ошибка: {str(root_error)}")
        print(f"Полный трейсбэк сохранен в файл: {CRASH_LOG_PATH}\n")
        
        with open(CRASH_LOG_PATH, "w", encoding="utf-8") as crash_file:
            crash_file.write(f"=== TUI CRASH SNAPSHOT: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            crash_file.write(f"Exception Type: {type(root_error).__name__}\n")
            crash_file.write(f"Exception Message: {str(root_error)}\n\n")
            crash_file.write("--- DETAILED TRACEBACK ---\n")
            traceback.print_exc(file=crash_file)
            
            # Если на момент падения сервер успел отдать кэш-карту — дампим её состояние
            try:
                crash_file.write("\n--- RECENT CACHE MAP STATE ---\n")
                crash_file.write(f"Total tokens total: {total_tokens if 'total_tokens' in locals() else 'unknown'}\n")
                if 'cache_map' in locals():
                    crash_file.write(f"Cache map items: {len(cache_map)}\n")
                    for idx, row in enumerate(cache_map):
                        crash_file.write(f"  Row {idx}: {row}\n")
            except Exception:
                pass
