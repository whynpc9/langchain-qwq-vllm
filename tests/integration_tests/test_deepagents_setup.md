# DeepAgents Integration Tests Setup

This document describes how to set up and run the DeepAgents integration tests for ChatQwenVllm.

## Prerequisites

### 1. Install Required Dependencies

```bash
# Install deepagents
pip install deepagents

# Optional: Install tavily for internet search tests
pip install tavily-python
```

### 2. Environment Configuration

Create a `.env` file in the project root with the following variables:

```bash
# VLLM Server Configuration
VLLM_API_BASE=http://localhost:8000/v1

# Optional: Tavily API Key for internet search tests
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: OpenAI API Key (if your VLLM server requires authentication)
OPENAI_API_KEY=your_vllm_api_key_here
```

### 3. Start VLLM Server

Before running the tests, start a VLLM server with a compatible Qwen model:

```bash
# Example: Start VLLM server with Qwen3-32B
vllm serve Qwen/Qwen3-32B-Instruct --port 8000

# Or with a different model
vllm serve Qwen/Qwen3-8B-Instruct --port 8000
```

## Running Tests

### Run All DeepAgents Tests

```bash
pytest tests/integration_tests/test_chat_models_with_deepagents.py -v
```

### Run Specific Test Categories

```bash
# Test basic deepagent functionality
pytest tests/integration_tests/test_chat_models_with_deepagents.py::TestChatQwenVllmWithDeepAgents::test_basic_deepagent_creation -v

# Test internet search integration (requires TAVILY_API_KEY)
pytest tests/integration_tests/test_chat_models_with_deepagents.py::TestChatQwenVllmWithDeepAgents::test_deepagent_with_internet_search -v

# Test async operations
pytest tests/integration_tests/test_chat_models_with_deepagents.py::TestChatQwenVllmWithDeepAgents::test_deepagent_async_operations -v

# Test compatibility requirements
pytest tests/integration_tests/test_chat_models_with_deepagents.py::TestDeepAgentsIntegrationRequirements -v
```

## Test Structure

The test suite includes:

### TestChatQwenVllmWithDeepAgents
Main test class with comprehensive deepagents integration tests:

- **test_basic_deepagent_creation**: Basic deepagent setup with simple calculator tool
- **test_deepagent_with_structured_output_tools**: Integration with structured output schemas
- **test_deepagent_with_internet_search**: Internet search using Tavily API
- **test_deepagent_with_file_operations**: Built-in file system tools
- **test_deepagent_with_subagents**: Subagent delegation functionality
- **test_deepagent_async_operations**: Async tool execution
- **test_deepagent_error_handling**: Error handling and resilience
- **test_deepagent_with_thinking_enabled**: ChatQwenVllm thinking capabilities

### TestDeepAgentsIntegrationRequirements
Compatibility and setup validation tests:

- **test_chatqwen_vllm_compatibility**: Verify required methods exist
- **test_deepagents_import_and_basic_setup**: Basic setup validation
- **test_environment_configuration**: Environment variable checks

## Example Usage

Here's how the deepagents integration works with ChatQwenVllm:

```python
from deepagents import create_deep_agent
from langchain_qwq.chat_models_vllm import ChatQwenVllm

# Initialize ChatQwenVllm
llm = ChatQwenVllm(
    model="Qwen/Qwen3-32B",
    temperature=0.1,
    enable_thinking=True,
)

# Define tools
def simple_calculator(operation: str, a: float, b: float) -> str:
    if operation == "add":
        return f"{a} + {b} = {a + b}"
    # ... other operations

# Create deep agent
agent = create_deep_agent(
    tools=[simple_calculator],
    instructions="You are an expert mathematician...",
    model=llm,
)

# Use the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "Calculate 15 * 8 + 23"}]
})
```

## Optional Features

### Internet Search with Tavily
Set `TAVILY_API_KEY` in your environment to enable internet search tests.

### Async Operations
The test suite includes async operation tests using `async_create_deep_agent`.

### Structured Output
Tests demonstrate integration between deepagents and ChatQwenVllm's structured output capabilities.

## Troubleshooting

### Common Issues

1. **VLLM Server Not Running**: Ensure VLLM server is started and accessible at the configured URL.

2. **Import Errors**: Install deepagents and optional dependencies:
   ```bash
   pip install deepagents tavily-python
   ```

3. **API Key Issues**: For Tavily tests, ensure `TAVILY_API_KEY` is set in your environment.

4. **Model Compatibility**: Ensure your VLLM server is running a compatible Qwen model.

### Skip Tests
Tests will automatically skip if required dependencies are not available:

- DeepAgents tests skip if `deepagents` is not installed
- Internet search tests skip if `tavily` is not installed or `TAVILY_API_KEY` is not set

## Performance Notes

- Tests use `temperature=0.1` for more deterministic responses
- `enable_thinking=True` is used to leverage ChatQwenVllm's reasoning capabilities
- Tests include timeout and retry mechanisms for robustness
