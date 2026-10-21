import os
import sys
from typing import List, Optional
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

# Инициализируем сервер под новую версию SDK v2
mcp = MCPServer("mcp-clarify")

@mcp.tool()
def ask_clarification(prompt: str, choices: Optional[List[str]] = None) -> str:
    """
    Use this tool when you encounter ambiguity, missing requirements, or need user confirmation.
    It stops the automatic execution loop and safely prompts the user for clarification in the CLI.
    All answers are automatically persisted to .goose-clarifications.md in the project root.
    """
    # Гарантируем, что стандартный вывод пуст перед интерактивным вводом
    sys.stdout.flush()
    
    # Красиво выводим вопрос в stderr, чтобы он отобразился в CLI у пользователя
    print(f"\n[CLARIFICATION REQUIRED] {prompt}", file=sys.stderr)
    if choices:
        print(f"Options: {', '.join(choices)}", file=sys.stderr)
    
    print("\nYour response: ", file=sys.stderr, end="")
    sys.stderr.flush()
    
    # Читаем ответ из терминала
    user_input = sys.stdin.readline().strip()
    
    # --- ЛОГИРОВАНИЕ НА ДИСК ---
    # os.getcwd() вернет корень проекта, откуда был запущен goose
    log_path = os.path.join(os.getcwd(), ".goose-clarifications.md")
    try:
        # Дописываем вопрос и ответ в аккуратном Markdown-формате
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"### Q: {prompt}\n")
            if choices:
                f.write(f"*Options:* {', '.join(choices)}\n")
            f.write(f"**A:** {user_input}\n\n")
    except Exception as e:
        print(f"[Warning] Could not write to log file: {e}", file=sys.stderr)
    # ----------------------------

    return f"User clarified: {user_input}"

if __name__ == "__main__":
    mcp.run()
