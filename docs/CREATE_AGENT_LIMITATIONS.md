# create_agent Compatibility And Limitations

This repository targets LangChain 1.x `create_agent()` usage.

## Environment prerequisites

- Python `>=3.10`
- LangChain `>=1.0.0`
- `langchain-openai >=1.0.0`
- `openai >=1.109.1`

Python 3.9 cannot install LangChain 1.x, so `create_agent()` validation must be run
under Python 3.10+.

## Validation script

Use:

```bash
PYTHONPATH=/tmp/langchain1py311:. python3.11 tests/scripts/validate_create_agent_endpoints.py
```

## Endpoint status as of 2026-03-18

### `old_vllm_qwen3_32b` (`legacy`, `Qwen/Qwen3-32B`)

- Supported:
  - `create_agent()` with tools
  - `create_agent()` with `ProviderStrategy`
  - `create_agent()` with `ToolStrategy + tools + enable_thinking=True`
- Limitation:
  - `ToolStrategy + tools + enable_thinking=False` returns `500`
  - `tools_only + enable_thinking=False` may return an empty final answer even when the
    invocation does not raise

### `new_vllm_qwen3_32b` (`modern`, `Qwen/Qwen3-32B`)

- Supported:
  - `create_agent()` with tools
  - `create_agent()` with `ProviderStrategy`
  - `create_agent()` with `ToolStrategy + tools + enable_thinking=True`
- Notes:
  - `ProviderStrategy` requires the client-side fallback in `ChatQwenVllm` because this
    endpoint may place JSON in `reasoning_content` instead of `content`
- Limitation:
  - `ToolStrategy + tools + enable_thinking=False` returns `500`
  - `tools_only + enable_thinking=False` may return an empty final answer even when the
    invocation does not raise

### `new_vllm_qwen3_5_27b` (`modern`, `Qwen/Qwen3.5-27B`)

- Supported:
  - `create_agent()` with tools
  - `create_agent()` with `ProviderStrategy`
  - `create_agent()` with `ToolStrategy + tools + enable_thinking=True`
  - `ToolStrategy + tools + enable_thinking=False`
  - `tools_only + enable_thinking=False`
- No known limitation from the validated combinations above
