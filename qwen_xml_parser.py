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

    def parse_chunk(self, chunk_text: str):
        """Accumulates incoming execution tokens and monitors function calling status flags."""
        self.raw_accumulated_text += chunk_text
        
        # Intercept the structural tag to confirm an active function invocation state
        if "<tool_call>" in self.raw_accumulated_text and not self.in_tool_call:
            self.in_tool_call = True
            # Attempt to pull out the extracted function execution target name
            func_match = re.search(r'<function=([^>]+)>', self.raw_accumulated_text)
            if func_match:
                self.tool_name = func_match.group(1).strip()
                
        return self.in_tool_call

    def extract_final_arguments(self) -> str:
        """
        Parses the fully accumulated token block text to extract a flat JSON string dictionary.
        Returns a guaranteed valid, stringified JSON parameter object schema for Goose.
        """
        if not self.tool_name:
            return "{}"
            
        # Extract parameter blocks bounded by <parameter=name>value</parameter> sequence wrappers
        param_matches = re.findall(r'<parameter=([^>]+)>(.*?)(?:</parameter>|$)', self.raw_accumulated_text, re.DOTALL)
        
        args_dict = {}
        for p_name, p_val in param_matches:
            clean_val = p_val.replace("</parameter>", "").strip()
            # Handle native casting for scalar integers while passing remainder targets as raw text strings
            if clean_val.isdigit():
                args_dict[p_name.strip()] = int(clean_val)
            else:
                args_dict[p_name.strip()] = clean_val
                
        # Return strict sanitized data sequence to prevent upstream mapping structure corruption
        return json.dumps(args_dict, ensure_ascii=False)
