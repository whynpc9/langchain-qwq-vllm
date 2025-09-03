# Integration Tests for ChatQwenVllm

This directory contains integration tests for the `ChatQwenVllm` class, which provides VLLM backend support for Qwen models.

## Test Structure

### TestChatQwenVllmIntegration
Standard LangChain integration tests that verify basic functionality:
- Basic invocation and streaming
- Tool calling capabilities
- Structured output support
- Async operations

### TestChatQwenVllmReasoningContent
Tests specifically for reasoning content handling:
- ✅ **Sync invocation**: `test_basic_invoke_with_reasoning()`
- ✅ **Sync streaming**: `test_stream_with_reasoning_content()`
- ✅ **Async invocation**: `test_async_invoke_with_reasoning()`
- ✅ **Async streaming**: `test_async_stream_with_reasoning_content()`
- ✅ **Reasoning structure**: Validates `reasoning_content` in `additional_kwargs`

### TestChatQwenVllmFeatures
Feature-specific tests:
- Tool calling functionality
- JSON mode structured output
- Batch processing (sync/async)
- `enable_thinking` parameter control
- Model configuration validation

### TestChatQwenVllmErrorHandling
Error handling and edge cases:
- Empty message handling
- Invalid model names

## Prerequisites

Before running the integration tests, you need:

1. **VLLM Server**: Start a VLLM server with a Qwen model
   ```bash
   # Example: Start VLLM server
   vllm serve Qwen/Qwen3-32B-Instruct --port 8000
   ```

2. **Environment Variables**: Configure the VLLM connection
   ```bash
   # Set VLLM API base URL (defaults to http://localhost:8000/v1)
   export VLLM_API_BASE="http://localhost:8000/v1"
   
   # VLLM servers typically don't require API keys, but if yours does:
   export OPENAI_API_KEY="your-vllm-api-key"
   ```

3. **Install Test Dependencies**:
   ```bash
   pip install -e ".[test]"
   ```

## Running Tests

### Run All VLLM Integration Tests
```bash
pytest tests/integration_tests/test_chat_models_vllm.py -v
```

### Run Specific Test Classes
```bash
# Test reasoning content handling
pytest tests/integration_tests/test_chat_models_vllm.py::TestChatQwenVllmReasoningContent -v

# Test core features
pytest tests/integration_tests/test_chat_models_vllm.py::TestChatQwenVllmFeatures -v

# Test error handling
pytest tests/integration_tests/test_chat_models_vllm.py::TestChatQwenVllmErrorHandling -v
```

### Run Specific Tests
```bash
# Test basic reasoning content access
pytest tests/integration_tests/test_chat_models_vllm.py::TestChatQwenVllmReasoningContent::test_basic_invoke_with_reasoning -v

# Test streaming with reasoning
pytest tests/integration_tests/test_chat_models_vllm.py::TestChatQwenVllmReasoningContent::test_stream_with_reasoning_content -v
```

## Test Configuration

The tests use the following default configuration:
- **Model**: `Qwen/Qwen3-32B`
- **Temperature**: 0.1
- **Max tokens**: 200 (for reasoning tests)
- **Enable thinking**: True
- **API Base**: `http://localhost:8000/v1` (configurable via `VLLM_API_BASE`)

## Expected Behavior

When properly configured, the tests validate:

1. **Reasoning Content Access**: 
   ```python
   response = llm.invoke("Solve: 2+2")
   reasoning = response.additional_kwargs.get("reasoning_content", "")
   assert "reasoning_content" in response.additional_kwargs
   ```

2. **Streaming with Reasoning**:
   ```python
   for chunk in llm.stream("Explain how to solve this"):
       if hasattr(chunk, 'additional_kwargs') and "reasoning_content" in chunk.additional_kwargs:
           # Handle reasoning content
           print(chunk.additional_kwargs["reasoning_content"], end="")
       elif hasattr(chunk, 'content') and chunk.content:
           # Handle response content
           print(chunk.content, end="")
   ```

3. **Async Operations**: All sync functionality also works in async mode.

## Troubleshooting

### Common Issues

1. **"The api_key client option must be set"**
   - VLLM servers usually don't require API keys
   - Try setting a dummy API key: `export OPENAI_API_KEY="dummy-key"`
   - Or modify your VLLM server to not require authentication

2. **Connection Errors**
   - Ensure VLLM server is running on the expected port
   - Check `VLLM_API_BASE` environment variable
   - Verify the model is loaded in VLLM

3. **Import Errors**
   - Install test dependencies: `pip install -e ".[test]"`
   - Ensure all required packages are available

## Note

These are integration tests that require an actual VLLM server to be running. If you don't have a VLLM server available, the tests will fail during the client initialization phase, which is expected behavior.
