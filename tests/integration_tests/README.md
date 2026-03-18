# Integration Validation

The old integration tests in this repository were removed because they encoded a
large number of hard-coded assumptions about specific local servers and no longer
matched the current compatibility target.

The meaningful integration validation now lives in these scripts:

- `tests/scripts/validate_vllm_qwen_endpoints.py`
  - raw model capability probes for thinking, structured output, and tool calling
- `tests/scripts/validate_create_agent_endpoints.py`
  - LangChain 1.x `create_agent()` compatibility matrix across the documented
    endpoints

## Recommended validation flow

1. Run unit tests:

```bash
python3 -m pytest tests/unit_tests/test_chat_models_vllm.py -q
```

2. Validate raw vLLM behavior:

```bash
python3 tests/scripts/validate_vllm_qwen_endpoints.py
```

3. Validate LangChain 1.x agent behavior under Python 3.10+:

```bash
PYTHONPATH=/tmp/langchain1py311:. python3.11 tests/scripts/validate_create_agent_endpoints.py
```
