# Local MLX Model Server for Goose & Qwen

A lightweight wrapper server built on top of `mlx_lm`, specifically designed for the **Goose AI Agent + Qwen (2.5/3.5/Coder)** stack running on Apple Silicon. 

This repository aims to save you time and tokens by solving typical local agent pain points (such as broken tool calls, runaway model reasoning loops, and VRAM thrashing).

Tested with `mlx_lm` version 0.31.3 and `Goose` version 1.45.0. 
This wrapper may also be useful for clients other than Goose or for other models.

---

## Why Use This (Instead of standard `mlx_lm.server`)

1. **KV-Cache Control & TurboQuant:**
   * Automatically forces 3-bit cache quantization (`bits = 3`, `quantized_start = 64`) to significantly save memory bandwidth.
   * **Isolated VRAM Memory Sessions:** Memory is split into two independent tracks—`agent` (primary multi-turn tool loops) and `utility` (background syntax checks, file listings, etc.). Background operations will never evict or corrupt your core agent's heavy context cache.
   * **Token-Level Cache Matching:** Longest Common Prefix (LCP) matching is done strictly via integer token IDs instead of raw strings. This prevents text words from breaking apart on string slicing boundaries.

2. **Reasoning Suppression (No Thinking Loops):**
   * Qwen models tend to get stuck in infinite reasoning loops inside `<thinking>` tags, bloating your context limits. This server strictly disables the thinking mode at the chat template argument level.

3. **Strict Sequential Execution (GPU Lock):**
   * Uses a robust `threading.Lock()` to ensure requests are processed **strictly one by one**. This is crucial for local machines with limited unified memory, preventing system slowdowns or out-of-memory (OOM) kernel panics.

4. **On-the-Fly XML Interception:**
   * Intercepts raw Qwen tool tags (`<tool_call>`, `<parameter>`) during streaming to keep the client UI completely clean. It then packages the final output into a standard OpenAI JSON format that Goose expects.

5. **Clean Transactional Logging:**
   * The terminal output is clean and visually split by `------` lines. It reports the precise total context size, as well as **Prefill Speed** (prompt evaluation) and **Decoding Speed** (token generation) in tokens per second.

---

## Repository Structure

* **`tq_server.py`** — The FastAPI entry point. Manages session cache registries, executes token-level LCP operations, and shifts hardware offsets for Apple Metal cache layers.
* **`stream_bridge.py`** — Multi-threaded orchestration. Runs the synchronous MLX generation inside a dedicated thread pool powered by **Starlette** (`starlette.concurrency.run_in_threadpool`). This safely offloads the heavy blocking GPU loop from FastAPI's main async event loop, applies the sequential GPU Lock, and pipes tokens via `asyncio.Queue`.
* **`goose_hooks.py`** — Request pre-processing. Normalizes payload layouts, converts descriptions into Hugging Face formats, and applies thinking suppression flags.
* **`qwen_xml_parser.py`** — A real-time token stream parser designed to intercept Qwen XML tool tags on the fly.
* **`response_formatters.py`** — Outputs clean SSE chunks and monolithic API responses. Forces accurate `prompt_len` attributes back to Goose so its internal token usage counters work properly.

---

## Getting Started

### 1. Installation

Install the server prerequisites

```bash
# Install core server dependencies
pip install fastapi uvicorn starlette
```

along with the verified working baseline versions of `mlx-lm` and the `goose` engine.

### 2. Usage

Run the backend server pointing to your local model path with optimized context boundaries and prefill step tuning:

```bash
python tq_server.py \
  --model ~/mlx_models/Qwen3.5-9B-MLX-4bit \
  --host 127.0.0.1 \
  --port 8080 \
  --max-tokens 4096 \
  --prefill-step-size 1024
```

### 3. Connecting to Goose

Configure your local Goose configuration profile to route execution tasks directly into the server engine by adding a custom provider endpoint pointing to `http://127.0.0.1:8080`.
