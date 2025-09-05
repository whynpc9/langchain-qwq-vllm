# langchain-qwq-vllm

A VLLM-optimized integration package for Qwen3 models with LangChain, forked from [@yigit353/langchain-qwq](https://github.com/yigit353/langchain-qwq). This implementation is specifically designed for VLLM deployments of Qwen3 models and explores integration with Deep Agents framework.

## ✨ Key Advantages

This fork provides essential enhancements for VLLM environments:

- **🔧 VLLM Native Support**: Optimized for VLLM backend with native `guided_json` support
- **🧠 Thinking + Structured Output**: Unique capability to combine reasoning mode with structured output (not available in original BaiLian platform)
- **🤖 Deep Agents Integration**: Full compatibility with Deep Agents framework for advanced AI workflows
- **⚡ Enhanced Performance**: Optimized for local/self-hosted VLLM deployments

## 🚀 Features

- **Streaming Support**: Real-time synchronous and asynchronous streaming
- **Reasoning Access**: Direct access to model's internal thinking process
- **Structured Output**: JSON schema-based structured responses using VLLM's `guided_json`
- **Tool Calling**: Function calling with parallel execution support
- **Deep Agents Integration**: Seamless integration with Deep Agents framework


## 🔧 Environment Setup

Configure your VLLM endpoint:

```bash
export VLLM_API_BASE="http://localhost:8000/v1"  # Default: http://localhost:8000/v1
```

Start your VLLM server:

```bash
vllm serve Qwen/Qwen3-32B --port 8000
```

## 💡 Basic Usage

### ChatQwenVllm

```python
from langchain_qwq import ChatQwenVllm

# Initialize the model
llm = ChatQwenVllm(
    model="Qwen/Qwen3-32B",
    temperature=0.1,
    max_tokens=2000,
    enable_thinking=True,  # Enable reasoning mode
)

# Basic interaction
response = llm.invoke("Explain quantum computing")
print(f"Response: {response.content}")

# Access reasoning process
reasoning = response.additional_kwargs.get("reasoning_content", "")
print(f"Model's thinking: {reasoning}")
```

### Streaming with Reasoning

```python
# Stream with thinking process
for chunk in llm.stream("Solve this math problem: 15 * 23 + 7"):
    if hasattr(chunk, 'additional_kwargs') and "reasoning_content" in chunk.additional_kwargs:
        print(f"🤔 {chunk.additional_kwargs['reasoning_content']}", end="")
    elif hasattr(chunk, 'content') and chunk.content:
        print(f"💬 {chunk.content}", end="")
```

### Structured Output with Reasoning

**🌟 This is the key advantage: combining thinking mode with structured output!**

```python
from pydantic import BaseModel, Field
from typing import List

class AnalysisResult(BaseModel):
    summary: str = Field(description="Brief analysis summary")
    key_points: List[str] = Field(description="Key findings")
    confidence: float = Field(description="Confidence score 0-1")

# Enable both thinking AND structured output
structured_llm = llm.with_structured_output(
    schema=AnalysisResult,
    method="json_schema"
)

result = structured_llm.invoke("Analyze the benefits of renewable energy")
print(f"Summary: {result.summary}")
print(f"Key Points: {result.key_points}")
print(f"Confidence: {result.confidence}")
```

### Tool Calling

```python
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions."""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except:
        return "Invalid expression"

# Bind tools to the model
llm_with_tools = llm.bind_tools([calculate])

response = llm_with_tools.invoke("What's 25 * 4 + 12?")
print(response.content)
```

### Combined: Tools + Structured Output + Reasoning

```python
class CalculationReport(BaseModel):
    problem: str = Field(description="Original math problem")
    steps: List[str] = Field(description="Solution steps")
    answer: float = Field(description="Final numerical answer")
    verification: str = Field(description="Verification of the result")

# This powerful combination is unique to our VLLM implementation
advanced_llm = llm.bind_tools([calculate]).with_structured_output(
    schema=CalculationReport,
    method="json_schema"
)

result = advanced_llm.invoke("Solve step by step: (15 + 8) * 3 - 7")
print(f"Problem: {result.problem}")
print(f"Steps: {result.steps}")
print(f"Answer: {result.answer}")
```

## 🤖 Deep Agents Integration

```python
from deepagents import create_deep_agent

# Create a research agent with web search capabilities
def web_search(query: str) -> str:
    """Search the web for information."""
    # Implementation depends on your search provider
    return f"Search results for: {query}"

agent = create_deep_agent(
    tools=[web_search],
    instructions="You are a research assistant with web access.",
    model=llm,  # Use our ChatQwenVllm model
)

# The agent can now use reasoning + tools + structured output
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Research the latest developments in quantum computing"
        }
    ]
})
```

## 📊 Advanced Examples

### Multi-step Analysis with File Operations

```python
from deepagents import create_deep_agent

def analyze_data(data_description: str) -> str:
    """Analyze data and provide insights."""
    return "Statistical analysis completed with key findings..."

# Agent with file system access
agent = create_deep_agent(
    tools=[analyze_data],
    instructions="You are a data analyst with file access.",
    model=llm,
    builtin_tools=["write_file", "read_file", "ls"],  # File system tools
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Analyze the sales data and create a report"
        }
    ],
    "files": {
        "sales_data.csv": "Q1,100000\nQ2,120000\nQ3,135000"
    }
})
```

### Async Operations

```python
import asyncio

async def process_requests():
    """Process multiple requests asynchronously."""
    
    tasks = [
        llm.ainvoke("Analyze renewable energy trends"),
        llm.ainvoke("Explain machine learning basics"),
        llm.ainvoke("Describe quantum computing applications")
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses, 1):
        print(f"Response {i}: {response.content[:100]}...")

# Run async processing
asyncio.run(process_requests())
```

## 🔄 Comparison with Original

| Feature | Original (BaiLian) | Our VLLM Fork |
|---------|-------------------|---------------|
| Thinking Mode | ✅ | ✅ |
| Structured Output | ✅ | ✅ |
| **Thinking + Structured** | ❌ | ✅ |
| Tool Calling | ✅ | ✅ |
| Deep Agents | ⚠️ Limited | ✅ Full Support |
| Local Deployment | ❌ | ✅ |
| Custom VLLM Optimizations | ❌ | ✅ |

## 📁 Project Structure

```
langchain-qwq-vllm/
├── langchain_qwq/
│   ├── chat_models.py          # Original ChatQwen classes
│   ├── chat_models_vllm.py     # Our VLLM-optimized ChatQwenVllm
│   └── utils.py                # Utility functions
├── tests/
│   ├── unit_tests/             # Unit tests
│   └── integration_tests/      # Deep Agents integration tests
├── docs/
│   ├── tool_calling_example.py  # Tool calling examples
│   └── vllm_structured_output_example.py  # Structured output examples
└── examples/
    └── tool_with_structured_output.py  # Combined examples
```

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
pytest tests/unit_tests/

# Integration tests (requires VLLM server)
pytest tests/integration_tests/

# Specific Deep Agents tests
pytest tests/integration_tests/test_chat_models_with_deepagents.py
```

## 🚀 Getting Started

1. **Start VLLM server**:
   ```bash
   vllm serve Qwen/Qwen3-32B --port 8000
   ```

2. **Install dependencies**:
   ```bash
   pip install langchain-qwq deepagents
   ```

3. **Run examples**:
   ```bash
   python docs/tool_calling_example.py
   python docs/vllm_structured_output_example.py
   ```

## 🤝 Contributing

This project focuses on VLLM optimization and Deep Agents integration. Contributions are welcome for:

- VLLM performance optimizations
- Deep Agents workflow examples
- Advanced reasoning + structured output use cases
- Documentation improvements

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Original Project**: [@yigit353/langchain-qwq](https://github.com/yigit353/langchain-qwq)
- **Development Tool**: This project was entirely developed using [Cursor](https://cursor.so/) AI-powered code editor
- **Framework**: Built on [LangChain](https://github.com/langchain-ai/langchain) and [VLLM](https://github.com/vllm-project/vllm)
- **AI Framework**: Integrated with [Deep Agents](https://github.com/multimodal-art-projection/deepagents)

---

**✨ Created with [Cursor](https://cursor.so/) - The AI-powered code editor**