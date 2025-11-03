# Integration Tests for LangChain 1.x

This directory contains integration tests for the langchain-qwq-vllm package with LangChain 1.x.

## Test Files

### `test_chat_models_vllm_langchain_agent.py`
Tests for LangChain 1.x agent integration with ChatQwenVllm:
- Basic agent creation and execution
- Tool calling with agents
- Error handling in agents
- Compatibility with `enable_thinking` parameter
- Agent execution with various configurations

### `test_structured_output_with_agent.py`
Tests for structured output support with LangChain 1.x agents:
- Simple Pydantic model extraction
- Complex nested structures
- Lists and arrays
- Optional fields
- Enums and literals
- Field validation
- Extraction accuracy
- Backward compatibility with `with_structured_output`

## Running Tests

### Run all integration tests:
```bash
pytest tests/integration_tests/ -v
```

### Run specific test file:
```bash
pytest tests/integration_tests/test_chat_models_vllm_langchain_agent.py -v
pytest tests/integration_tests/test_structured_output_with_agent.py -v
```

### Run specific test:
```bash
pytest tests/integration_tests/test_chat_models_vllm_langchain_agent.py::TestChatQwenVllmLangChainAgent::test_basic_agent_creation -v
```

## Prerequisites

1. **VLLM Server Running**: Ensure a VLLM server is running with Qwen3 model:
   ```bash
   vllm serve Qwen/Qwen3-32B --port 8000
   ```

2. **Environment Variables**: Set the VLLM API base:
   ```bash
   export VLLM_API_BASE="http://localhost:8000/v1"
   export OPENAI_API_KEY="dummy-key-for-vllm"  # Required but can be dummy
   ```

3. **Dependencies**: Install test dependencies:
   ```bash
   pip install -e ".[test]"
   ```

## Test Coverage

### Agent Integration (test_chat_models_vllm_langchain_agent.py)
- ✅ LangChain version verification
- ✅ `create_agent` availability check
- ✅ Basic agent creation with tools
- ✅ Agent execution with simple queries
- ✅ Error handling in agent loops
- ✅ `enable_thinking` compatibility with agents

### Structured Output (test_structured_output_with_agent.py)
- ✅ Simple field extraction (ContactInfo)
- ✅ Nested structures (Person with Address)
- ✅ Lists and complex types (ProductReview)
- ✅ Optional fields (EventInfo)
- ✅ Enums and literals (Task with Priority)
- ✅ Field validation (OrderInfo)
- ✅ Extraction accuracy (MovieInfo)
- ✅ Backward compatibility (`with_structured_output`)

## Migration Notes

These tests are designed for **LangChain 1.x** and use the modern agent framework:
- Uses `create_agent()` instead of `DeepAgent`
- Uses `ProviderStrategy` for structured output with agents
- Compatible with LangGraph's `CompiledStateGraph`
- Removed deprecated LangChain 0.3 patterns

## Known Limitations

1. **Structured Output with Agents**: Must use explicit `ProviderStrategy` as ChatQwenVllm is not automatically recognized by LangChain's `_supports_provider_strategy` function.

2. **VLLM Constraints**: 
   - Cannot use `guided_json` with `enable_thinking` simultaneously
   - Cannot use `guided_json` with `tools` simultaneously

3. **Test Requirements**: All tests require a running VLLM server with appropriate models loaded.
