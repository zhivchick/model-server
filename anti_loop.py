import re
import json
import logging

logger = logging.getLogger("mlx_lm_server.antiloop")

# 🎨 COLORS FOR TERM
C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_RED    = "\033[91m"
C_YELLOW = "\033[93m"

class AntiLoopEngine:
    """Manages consecutive repetition blocks and triggers dynamic Goose Compact signals when blind."""
    def __init__(self):
        self.last_tool = None
        self.last_skeleton = None
        self.hit_count = 0

    def _build_argument_skeleton(self, args_dict: dict) -> str:
        raw_str = json.dumps(args_dict, ensure_ascii=False)
        return re.sub(r'\d+', '', raw_str)

    def evaluate_and_process(self, full_text: str, parser_tool_name: str, extracted_args_dict: dict) -> tuple:
        """
        Evaluates ongoing tool sequences. 
        Returns (tool_name, tool_args) or (None, warning_text) to force pull the brake.
        """
        func_match = re.search(r'<?function=([^>]+)>?', full_text)
        if func_match:
            tool_name = func_match.group(1).strip().replace('"', '').replace("'", "")
        else:
            tool_name = parser_tool_name if parser_tool_name else "shell"

        current_skeleton = self._build_argument_skeleton(extracted_args_dict)
        final_json_args = json.dumps(extracted_args_dict, ensure_ascii=False)

        if self.last_tool == tool_name and self.last_skeleton == current_skeleton:
            self.hit_count += 1
            logger.warning(f"{C_YELLOW}⚠️ [ANTI-LOOP FIREWALL] Repetitive pattern found! Tool: '{tool_name}', Depth: {self.hit_count}{C_RESET}")
            
            # 🎯 STAGE 2: EMERGENCY DIALOGUE BRAKE (4th identical attempt)
            # The model is officially blind at 70k+ context. Stop sending shell errors, drop tool_chain!
            if self.hit_count >= 4:
                logger.error(f"{C_BOLD}{C_RED}🚨 [CONTEXT EMERGENCY] Hard threshold reached. Forcing Dialogue Brake to trigger Goose Compact!{C_RESET}")
                
                # Flush state to prepare for post-compact execution maps
                self.hit_count = 0
                self.last_tool = None
                self.last_skeleton = None
                
                compact_trigger_text = (
                    "⚠️ [SERVER NOTICE] Attention window degradation detected due to context scale. "
                    "Forcing thread synchronization break to trigger active memory compacting routines."
                )
                # 🎯 Return None to notify stream_bridge about a forced text fallback stop
                return None, compact_trigger_text

            # STAGE 1: Standard continuous pressure via shell exit 1 (attempts 2 and 3)
            if self.hit_count >= 2:
                logger.error(f"{C_BOLD}{C_RED}🚨 [FIREWALL BRICKWALL] Continuous loop lock sustained on tool '{tool_name}'. Deflecting payload.{C_RESET}")
                forced_tool_name = "shell"
                payload_warning_text = (
                    f"echo 'Execution Error: The tool \"{tool_name}\" is locked in an infinite repetition loop. "
                    f"Action blocked by firewalled server framework context constraints. Change your arguments "
                    f"or tool usage strategy completely to proceed.' && exit 1"
                )
                return forced_tool_name, json.dumps({"command": payload_warning_text}, ensure_ascii=False)
        else:
            if self.hit_count > 0:
                logger.info("🎉 [LOOP BROKEN] Model successfully pivoted to a different execution strategy. Flushing firewall blocks.")
            self.last_tool = tool_name
            self.last_skeleton = current_skeleton
            self.hit_count = 0

        logger.info(f"Generated Tool Call: '{tool_name}' with args: {final_json_args}")
        return tool_name, final_json_args

# Global singleton
anti_loop_engine = AntiLoopEngine()
