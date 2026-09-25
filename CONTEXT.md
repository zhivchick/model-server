# Model Server: Context & Architecture Guide

This document maintains architectural context, component responsibilities, and operational notes for the **Local MLX Model Server for Goose & Qwen**.

---

## 1. Project Goal & Ecosystem

- **Target Hardware**: Apple Silicon Macs (Unified Memory, Metal GPU).
- **Core Engine**: `mlx_lm` (tested on 0.31.x).
- **Primary Client**: Goose AI Agent (v1.45+), Cursor, Cline, Claude Code, or OpenAI-compatible CLI tools.
- **Target Models**: Qwen 2.5 / Qwen 3.5 / Qwen 2.5-Coder (MLX 4-bit, 3-bit, or TurboQuant).

---

## 2. Component Directory & Responsibilities

| File | Primary Responsibility | Critical Nuance |
| :--- | :--- | :--- |
| **`tq_server.py`** | FastAPI endpoint (`/v1/chat/completions`, `/v1/context/*`), dual session cache registries, LCP token-level matching. | Manages isolated Metal KV-caches for `agent` and `utility` tracks. Must release `ASYNC_SERVER_LOCK` on client disconnect. |
| **`stream_bridge.py`** | Dedicated Starlette threadpool runner (`sync_generation_worker`), token streaming to `asyncio.Queue`, on-the-fly XML/JSON interception. | Must pass OpenAI `tool_calls` structure when Qwen XML tags are parsed in normal flow. |
| **`anti_loop.py`** | Repetitive pattern firewall, idle edit detection, 3-tier defense escalation (tool error -> user intervention -> dialogue brake). | Protects GitHub entity IDs (`gh issue view 24` vs `25`) from digit stripping. Injects simulated user intervention on calls 4 and 5 before hard brake on call 6. |
| **`cache_monitor.py`** | Live curses TUI context monitor (`/v1/context/raw`), regex file extractor (`cat`, `view_file`, `sed`), token weight calculation. | Must safely handle row truncation badges and curses terminal resize events. |
| **`qwen_xml_parser.py`** | Real-time XML streaming tag parser (`<tool_call>`, `<function=...>`, `<parameter=...>`). | Converts tag attributes and body into valid stringified JSON dictionary. |
| **`goose_hooks.py`** | Normalizes OpenAI vs HuggingFace tool schemas, suppresses thinking loops (`enable_thinking: false`). | Prevents infinite `<thinking>` loops from consuming the prompt context window. |
| **`context_pipeline.py`**| Freezes `<current-time>` and `<compaction>` tags into static tokens before tokenizer runs. | Prevents KV-cache invalidation caused by dynamic wall-clock timestamps. |
| **`perf_tracker.py`** | Prefill and decoding speed metrics (t/s), peak speed tracking, moving averages. | Filters out micro-requests (<512 prompt tokens, <25 completion tokens) from averages. |
| **`goose_killer.py`** | Watchdog invoking ACP / REST endpoint to interrupt runaway Goose agent loops and trigger compact. | Reads `ACP_SERVER_URL` and `GOOSE_SECRET_KEY` from environment variables. |
| **`response_formatters.py`** | SSE chunks (`data: {...}\n\n`) and monolithic OpenAI JSON responses with usage metadata. | Supplies accurate `prompt_tokens` so client token trackers remain in sync. |

---

## 3. Key Defensive Invariants

1. **Hardware Time Lock**: Dynamic clock timestamps (`<current-time>`) must be replaced with static tokens *before* tokenization. If not, every agent iteration causes a 100% KV-cache miss.
2. **Sequential GPU Lock**: Local Unified Memory cannot survive concurrent heavy LLM prefill steps. `ASYNC_SERVER_LOCK` enforces strictly 1 request at a time.
3. **Idle Edit Defense**: Models in long context loops often issue identical `before` and `after` string replacements. The firewall immediately catches and deflects this on depth 0.
4. **Tool Call Emission**: When Qwen XML is parsed during streaming, the raw tags are suppressed from terminal output, and the final chunk MUST carry `finish_reason="tool_calls"` with valid `tool_name` and `arguments`.
5. **Anti-Loop Escalation Tiers**: Tier 1 (calls 2 & 3 / hit 1 & 2) deflects with tool errors (`echo ... && exit 1`); Tier 2 (calls 4 & 5 / hit 3 & 4) escalates to human operator intervention prompts (`[USER INTERVENTION]` and `[USER DIRECTIVE]`); Tier 3 (call 6 / hit 5+) forces a hard dialogue brake returning control to the human. GitHub entity IDs are preserved so issue/PR browsing is never falsely flagged.
