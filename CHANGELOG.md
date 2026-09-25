# Model Server Changelog

All notable changes, fixes, and context notes are recorded here to track the evolution of the codebase.

---

## [Anti-Loop Escalation & GitHub ID Exemption] - 2026-09-24

### 1. `anti_loop.py`
- **Feature (GitHub ID Exemption)**: Added `_protect_github_entity_ids` to translate numbers in GitHub entity commands (`gh issue view 24`, `gh pr diff 25`, `gh run view #123`, `gh api .../issues/<id>`) into non-digit token markers before skeleton digit stripping. This prevents consecutive issue/PR/run browsing from being falsely flagged as repetitive loops, while continuing to catch pagination limits (`-L 10` vs `-L 20`) and sliding file reads (`head -10` vs `head -11`, `sed -n '1,10p'`).
- **Feature (3-Tier Anti-Loop Escalation)**:
  - **Tier 1 (Calls 2 & 3 / Hits 1 & 2)**: Tool error deflection with actionable Smart Assist hints via `shell` `echo ... && exit 1`.
  - **Tier 2 (Calls 4 & 5 / Hits 3 & 4)**: Simulated Human Operator Interventions (`[USER INTERVENTION]` and `[USER DIRECTIVE - FINAL WARNING]`). Model receives explicit user directives both at prompt level and tool output level instructing it to stop repeating the command.
  - **Tier 3 (Call 6 / Hit 5+)**: Hard Dialogue Brake (`t_name is None`) with terminal session pause, returning control to human user and resetting loop state.

### 2. `goose_hooks.py`
- **Feature (Prompt-Level User Intervention)**: When `anti_loop_engine.hit_count` is 2 or 3 (prior to calls 4 and 5), dynamically appends a `user` role message to the Jinja chat context informing the model directly from the user to halt the repetition loop and choose an alternative path.

### 3. `test_anti_loop.py`
- **Test Suite**: Added unit test coverage for GitHub entity ID exemption, sliding window loop detection, 3-tier escalation progression, and pivot counter resets.

---

## [Static Analysis & Hardening Pass] - 2026-09-12

### 1. `cache_monitor.py`
- **Bug Fixed**: Line 245 was adding a raw tuple `processed_rows[i]` to integer `truncated_tokens_sum` (`0 + ("file.py", 120, ...)`), causing a fatal `TypeError: unsupported operand type(s) for +=: 'int' and 'tuple'` as soon as terminal height truncated visible rows.
- **Fix**: Changed to `truncated_tokens_sum += processed_rows[i][1]`.
- **Impact**: TUI monitor can now safely resize to small terminal windows without crashing.

### 2. `tq_server.py`
- **Bug Fixed**: `LAST_LIVE_RAW_TEXT` was declared `global` inside `_render_and_tokenize` and `get_raw_context`, but never initialized at the module level. If `cache_monitor` polled `/v1/context/raw` before the first inference request completed, a `NameError` crashed the endpoint.
- **Fix**: Initialized `LAST_LIVE_RAW_TEXT = ""` at module level.
- **Bug Fixed**: `ASYNC_SERVER_LOCK.release()` was called after `async for chunk in async_queue_bridge`. If client cancelled/disconnected mid-stream, `StreamingResponse` was aborted, bypassing the release line and permanently deadlocking the server for all subsequent requests.
- **Fix**: Wrapped stream consumption in `try ... finally` to guarantee lock release upon completion, error, or client cancellation.
- **Enhancement**: Implemented active logging in `_print_pipeline_telemetry` stub (`req_id`, `target`, `prompt_len`).

### 3. `stream_bridge.py`
- **Bug Fixed**: In line 217 (normal scenario where `t_name == tool_invocation_name` for XML tools), the bridge emitted `build_streaming_chunk(..., finish_reason="stop")` without `tool_name` or `tool_args`. Because XML tags were filtered during generation, Goose never received any tool call payload.
- **Fix**: Corrected to pass `tool_name=t_name, tool_args=t_args, finish_reason="tool_calls"`.
- **Impact**: Goose now properly receives the converted OpenAI tool_call JSON from Qwen's XML stream.

### 4. `anti_loop.py`
- **Bug Fixed**: Line 97 used an inverted ternary `if not forced_tool_name == "shell"` which evaluated to the `else` branch when `forced_tool_name == "shell"`. This returned unquoted human text directly into the bash `"command"`, causing `bash: Execution: command not found`.
- **Fix**: Cleanly wrapped all deflection payloads in `echo '{safe_text}' && exit 1` with single-quote escaping.

### 5. `goose_killer.py`
- **Enhancement**: Added `os.getenv("ACP_SERVER_URL")` and `os.getenv("GOOSE_SECRET_KEY")` fallbacks to allow zero-code configuration via environment variables.
