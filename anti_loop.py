import re
import json
import logging

logger = logging.getLogger("mlx_lm_server.antiloop")

class AntiLoopEngine:
    """Manages consecutive repetition blocks and forces execution fault blocks to halt agent freezes."""
    def __init__(self):
        self.last_tool = None
        self.last_args_hash = None
        self.hit_count = 0

    def evaluate_and_process(self, full_text: str, parser_tool_name: str, extracted_args_dict: dict) -> tuple:
        """Evaluates ongoing tool sequences and returns an active firewall override payload if triggered."""
        # Detect tool name via fallback full-text scan
        func_match = re.search(r'<function=([^>]+)>', full_text)
        tool_name = func_match.group(1).strip() if func_match else (parser_tool_name if parser_tool_name else "shell")

        final_json_args = json.dumps(extracted_args_dict, ensure_ascii=False)
        current_args_hash = hash(final_json_args)

        if self.last_tool == tool_name and self.last_args_hash == current_args_hash:
            self.hit_count += 1
            logger.warning(f"⚠️ [ANTI-LOOP INFRASTRUCTURE] Repetitive call signature found. Depth count: {self.hit_count}")
            
            # If threshold hit, apply sustained pressure without resetting the counter registers
            if self.hit_count >= 2:
                logger.error(f"🚨 [FIREWALL BRICKWALL] Repetitive loop lock sustained on tool '{tool_name}'. Returning fault injection.")
                forced_tool_name = "shell"
                payload_warning_text = (
                    f"echo 'Execution Error: The tool \"{tool_name}\" is locked in an infinite repetition loop. "
                    f"Action blocked by firewalled server framework context constraints. "
                    f"Do not attempt to re-run this exact layout configuration again. Change your arguments, "
                    f"target parameters, or tool routing strategy to proceed.' && exit 1"
                )
                return forced_tool_name, json.dumps({"command": payload_warning_text}, ensure_ascii=False)
        else:
            if self.hit_count > 0:
                logger.info("🎉 [LOOP BROKEN] Model successfully pivoted to a different execution strategy. Flushing firewall blocks.")
            self.last_tool = tool_name
            self.last_args_hash = current_args_hash
            self.hit_count = 0

        logger.info(f"Generated Tool Call: '{tool_name}' with args: {final_json_args}")
        return tool_name, final_json_args

# Global singleton
anti_loop_engine = AntiLoopEngine()
