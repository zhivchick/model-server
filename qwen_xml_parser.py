import re
import json

class QwenXmlParser:
    """
    Isolates the parsing logic for Qwen's native XML tool tags.
    Converts raw XML tag streams into valid OpenAI JSON argument objects.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.in_tool_call = False
        self.tool_name = ""
        self.raw_accumulated_text = ""

    def parse_chunk(self, chunk_text: str) -> bool:
        """Accumulates incoming execution tokens and monitors function calling status flags."""
        self.raw_accumulated_text += chunk_text
        
        if "<tool_call>" in self.raw_accumulated_text and not self.in_tool_call:
            self.in_tool_call = True
            
        if self.in_tool_call and not self.tool_name:
            func_match = re.search(r'<function=([^>]+)>', self.raw_accumulated_text)
            if func_match:
                self.tool_name = func_match.group(1).strip().strip("'\"")
                
        return self.in_tool_call

    def extract_arguments_dict(self) -> dict:
        """
        Единый метод извлечения аргументов из XML-вызова Qwen 3.5.
        Возвращает готовый словарь аргументов без разрушения строковых ID с лидирующими нулями.
        """
        param_matches = re.findall(
            r'<parameter=([^>]+)>(.*?)(?:</parameter>|$)',
            self.raw_accumulated_text,
            re.DOTALL
        )
        
        args_dict = {}
        for p_name, p_val in param_matches:
            key = p_name.strip()
            clean_val = p_val.replace("</parameter>", "").strip()
            
            # Конвертируем только чистые целые числа без лидирующих нулей (например, line_number = 42, но не "0012")
            if re.fullmatch(r'-?(?:0|[1-9]\d*)', clean_val):
                try:
                    args_dict[key] = int(clean_val)
                    continue
                except ValueError:
                    pass
            elif clean_val in ("true", "false"):
                args_dict[key] = (clean_val == "true")
                continue

            args_dict[key] = clean_val
            
        return args_dict
