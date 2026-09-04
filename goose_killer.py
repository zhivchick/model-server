import httpx
import logging

logger = logging.getLogger("mlx_lm_server.killer")

# Настройки подключения к твоему TLS-серверу Goose
ACP_SERVER_URL = "https://127.0.0.1:3000"
SECRET_KEY = "YOUR_SECRET"

async def trigger_goose_compaction_break(session_id: str):
    """
    🎯 ИНТЕЛЛЕГЕНТНЫЙ ВЫЗОВ СЕССИОННОГО REST API GOOSE
    Шлёт авторизованный POST-запрос на сброс активного Chaining-цикла
    """
    if not session_id:
        logger.warning("⚠️ [Watchdog Triggered] Передан пустой ID сессии!")
        return False
        
    connection_headers = {
        "Authorization": f"Bearer {SECRET_KEY}",
        "Content-Type": "application/json"
    }
    
    # Передаем ключ сессии в теле запроса согласно спецификации ACP REST
    payload = {
        "session_id": session_id
    }
    
    # Мы поочередно пробуем официальные ACP пути для прерывания цепочки
    endpoints = [
        f"{ACP_SERVER_URL}/v1/sessions/reset",
        f"{ACP_SERVER_URL}/v1/session/interrupt",
        f"{ACP_SERVER_URL}/api/session/{session_id}/interrupt"
    ]
    
    async with httpx.AsyncClient(verify=False) as client:
        for url in endpoints:
            try:
                response = await client.post(url, headers=connection_headers, json=payload, timeout=2.0)
                
                # Коды 200 или 204 означают успешное прерывание выполнения
                if response.status_code in [200, 204]:
                    logger.warning(f"💥 [💥 WATCHDOG REST SUCCESS] Цепочка Goose прервана через эндпоинт: {url}!")
                    
                    # Сразу следом шлём команду на автоматический Компакт истории!
                    compact_url = f"{ACP_SERVER_URL}/v1/sessions/compact"
                    await client.post(compact_url, headers=connection_headers, json=payload, timeout=2.0)
                    logger.warning("🧹 [Watchdog REST Summary] Команда на принудительный Compact успешно отправлена!")
                    return True
                    
                elif response.status_code == 401:
                    logger.debug(f"ℹ️ [Watchdog REST] Путь {url} вернул 401 (промах авторизации).")
                elif response.status_code == 404:
                    logger.debug(f"ℹ️ [Watchdog REST] Путь {url} вернул 404 (нет такого эндпоинта).")
                    
            except Exception as e:
                logger.debug(f"ℹ️ Ошибка запроса к {url}: {str(e)}")
                
    logger.error("💀 [Watchdog Failure] Не удалось достучаться до сессионного REST API Goose. Все пути вернули ошибки.")
    return False
