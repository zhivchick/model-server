import logging

logger = logging.getLogger("mlx_lm_server.telemetry")

# 🎨 TERMINAL ANSI COLOR CODES
C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_GREY   = "\033[90m"
C_RED    = "\033[91m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN   = "\033[96m"

class PerformanceTracker:
    """Handles global state and historical logging matrices for MLX execution speeds with smart noise filtering."""
    def __init__(self):
        self.prefill_speeds = []
        self.prefill_peak = 0.0
        self.decoding_speeds = []
        self.decoding_peak = 0.0

    def record_metrics(self, current_prefill: float, current_decode: float, total_context_len: int, prompt_chunk_len: int, completion_len: int):
        """Updates internal statistics tables, filtering out micro-requests and rendering colored telemetry."""
        
        # 📊 PREFILL TELEMETRY FILTER
        if current_prefill > 0:
            if current_prefill > self.prefill_peak:
                self.prefill_peak = current_prefill
            if prompt_chunk_len >= 512:
                self.prefill_speeds.append(current_prefill)
                
        # 📊 DECODING TELEMETRY FILTER
        if current_decode > 0:
            if current_decode > self.decoding_peak:
                self.decoding_peak = current_decode
            if completion_len >= 25:
                self.decoding_speeds.append(current_decode)

        avg_prefill = sum(self.prefill_speeds) / len(self.prefill_speeds) if self.prefill_speeds else 0.0
        avg_decode = sum(self.decoding_speeds) / len(self.decoding_speeds) if self.decoding_speeds else 0.0

        logger.info(f"[TRACKER METRICS] Reported context size to Goose: {total_context_len} tokens.")
        
        # Determine performance color status compared to history
        p_color = C_GREEN if current_prefill >= avg_prefill else C_YELLOW if current_prefill > (avg_prefill * 0.75) else C_RED
        d_color = C_GREEN if current_decode >= avg_decode else C_YELLOW if current_decode > (avg_decode * 0.75) else C_RED

        # Print beautifully formatted, ultra-scannable colored matrix
        print(f"  Phase     |  Current Speed  |  Average Speed  | Peak Speed ")
        print(f"  Prefill   | {p_color}{current_prefill:11.2f} t/s{C_RESET} | {C_CYAN}{avg_prefill:11.2f} t/s{C_RESET} | {C_BOLD}{C_GREEN}{self.prefill_peak:10.2f} t/s{C_RESET}")
        print(f"  Decoding  | {d_color}{current_decode:11.2f} t/s{C_RESET} | {C_CYAN}{avg_decode:11.2f} t/s{C_RESET} | {C_BOLD}{C_GREEN}{self.decoding_peak:10.2f} t/s{C_RESET}")

# Global singleton
perf_tracker = PerformanceTracker()
