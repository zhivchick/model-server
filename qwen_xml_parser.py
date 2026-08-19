# qwen_xml_parser.py
import re
import json

class QwenXmlParser:
    """
    Класс для изоляции логики парсинга нативных XML-тегов инструментов Qwen.
    Превращает XML-структуру в валидный OpenAI JSON формат аргументов.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.in_tool_call = False
        self.tool_name = ""
        self.raw_accumulated_text = ""

    def parse_chunk(self, chunk_text: str):
        """Накапливает входящие токены и проверяет статус вызова функции."""
        self.raw_accumulated_text += chunk_text
        
        # Если поймали открывающий тег, значит пошел вызов инструмента
        if "<tool_call>" in self.raw_accumulated_text and not self.in_tool_call:
            self.in_tool_call = True
            # Пытаемся вытащить имя функции
            func_match = re.search(r'<function=([^>]+)>', self.raw_accumulated_text)
            if func_match:
                self.tool_name = func_match.group(1).strip()
                
        return self.in_tool_call

    def extract_final_arguments(self) -> str:
        """
        Парсит накопленный XML текст и собирает из него плоский JSON-словарь аргументов.
        Возвращает готовую валидную JSON строку.
        """
        if not self.tool_name:
            return "{}"
            
        # Ищем все пары тегов <parameter=имя>значение</parameter>
        param_matches = re.findall(r'<parameter=([^>]+)>(.*?)(?:</parameter>|$)', self.raw_accumulated_text, re.DOTALL)
        
        args_dict = {}
        for p_name, p_val in param_matches:
            clean_val = p_val.replace("</parameter>", "").strip()
            # Пытаемся сохранить числа как числа, остальное как строки
            if clean_val.isdigit():
                args_dict[p_name.strip()] = int(clean_val)
            else:
                args_dict[p_name.strip()] = clean_val
                
        # Возвращаем гарантированно валидную JSON-строку для Goose
        return json.dumps(args_dict, ensure_ascii=False)
