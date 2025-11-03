# langchain-qwq-vllm

A VLLM-optimized integration package for Qwen3 models with **LangChain 1.0**, forked from [@yigit353/langchain-qwq](https://github.com/yigit353/langchain-qwq). This implementation is specifically designed for VLLM deployments of Qwen3 models and uses the modern LangChain 1.x agent framework.

> **📌 Version Notice**: This is the **LangChain 1.0** version (v1.0.0+). For the legacy LangChain 0.3 version, please check out the [`classic`](https://github.com/yourusername/langchain-qwq-vllm/tree/classic) branch.

## ✨ Key Advantages

This fork provides essential enhancements for VLLM environments:

- **🔧 VLLM Native Support**: Optimized for VLLM backend with native `guided_json` support
- **🧠 Thinking + Structured Output**: Unique capability to combine reasoning mode with structured output (not available in original BaiLian platform)
- **🤖 LangChain 1.0 Agents**: Full compatibility with modern LangChain 1.x agent framework using `create_agent()`
- **📋 Provider Strategy**: Native structured output support for LangChain 1.0 agents
- **⚡ Enhanced Performance**: Optimized for local/self-hosted VLLM deployments

### What's New in v1.0.0

- ✅ Migrated to LangChain 1.0+ (langchain-core, langchain, langgraph)
- ✅ Replaced DeepAgent with standard `create_agent()` API
- ✅ Implemented Provider Strategy for structured output with agents
- ✅ Added comprehensive integration tests for LangChain 1.x
- ✅ Improved documentation and examples

For migration guide from 0.x to 1.0, see [LANGCHAIN_V1_MIGRATION.md](LANGCHAIN_V1_MIGRATION.md).

## 🚀 Features

- **LangChain 1.0 Ready**: Fully compatible with LangChain 1.x, LangGraph 1.x
- **Streaming Support**: Real-time synchronous and asynchronous streaming
- **Reasoning Access**: Direct access to model's internal thinking process with `enable_thinking`
- **Structured Output**: Provider Strategy with VLLM's native `guided_json` support
- **Tool Calling**: Function calling with parallel execution support
- **Modern Agent Framework**: Full support for LangChain 1.0's `create_agent()` pattern
- **Comprehensive Tests**: 19 integration tests covering all features


## 📦 Installation

### For LangChain 1.0 (Current Version)

```bash
# Install from main branch
pip install git+https://github.com/yourusername/langchain-qwq-vllm.git

# Or clone and install
git clone https://github.com/yourusername/langchain-qwq-vllm.git
cd langchain-qwq-vllm
pip install -e .
```

### For LangChain 0.3 (Legacy)

```bash
# Install from classic branch
pip install git+https://github.com/yourusername/langchain-qwq-vllm.git@classic

# Or clone and install
git clone -b classic https://github.com/yourusername/langchain-qwq-vllm.git
cd langchain-qwq-vllm
pip install -e .
```

## 🔧 Environment Setup

Configure your VLLM endpoint:

```bash
export VLLM_API_BASE="http://localhost:8000/v1"  # Default: http://localhost:8000/v1
export OPENAI_API_KEY="dummy-key-for-vllm"      # Required but can be dummy value
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

### Structured Output with Agents (LangChain 1.0)

**🌟 New in v1.0: Native Provider Strategy support for structured output!**

```python
from pydantic import BaseModel, Field
from typing import List
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy

class AnalysisResult(BaseModel):
    summary: str = Field(description="Brief analysis summary")
    key_points: List[str] = Field(description="Key findings")
    confidence: float = Field(description="Confidence score 0-1")

# Create agent with structured output using ProviderStrategy
agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are an expert analyst.",
    response_format=ProviderStrategy(AnalysisResult)  # Important: Use ProviderStrategy!
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze the benefits of renewable energy"}]
})

analysis = result["structured_response"]
print(f"Summary: {analysis.summary}")
print(f"Key Points: {analysis.key_points}")
print(f"Confidence: {analysis.confidence}")
```

> **⚠️ Important**: Always use `ProviderStrategy(schema)` explicitly for ChatQwenVllm with agents to ensure VLLM's native `guided_json` is used.

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

## 🤖 LangChain 1.0 Agent Integration

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Implementation depends on your search provider
    return f"Search results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return str(eval(expression))

# Create an agent using LangChain 1.0 pattern
agent = create_agent(
    model=llm,  # Use our ChatQwenVllm model
    tools=[web_search, calculator],
    system_prompt="You are a research assistant with web access and calculation abilities."
)

# The agent can now use reasoning + tools
result = agent.invoke({
    "messages": [
        HumanMessage(content="Research the latest developments in quantum computing and calculate its market size growth rate")
    ]
})

# Access the results
final_message = result["messages"][-1]
print(final_message.content)

# View all messages including tool calls
for msg in result["messages"]:
    print(f"{msg.__class__.__name__}: {msg.content[:100]}...")
```

### Agent with Structured Output

Combine agent tool calling with structured output:

```python
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ProviderStrategy

class ResearchReport(BaseModel):
    topic: str = Field(description="Research topic")
    summary: str = Field(description="Executive summary")
    key_findings: list[str] = Field(description="Main findings")
    sources: list[str] = Field(description="Information sources used")

# Agent that returns structured research report
agent = create_agent(
    model=llm,
    tools=[web_search, calculator],
    system_prompt="You are a research assistant. Use tools to gather information, then provide a structured report.",
    response_format=ProviderStrategy(ResearchReport)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Research AI market trends in 2024"}]
})

report = result["structured_response"]
print(f"Topic: {report.topic}")
print(f"Summary: {report.summary}")
print(f"Key Findings: {report.key_findings}")
```

> **⚠️ Note**: When using `response_format` with agents, tools cannot be called in the final response. For tool calling + structured output, process in multiple steps or use `with_structured_output()` on the model directly.

## ⚠️ Important Limitations

### Structured Output Constraints

Due to VLLM's implementation, the following combinations are **not supported**:

```python
# ❌ Cannot use guided_json with enable_thinking
llm = ChatQwenVllm(enable_thinking=True)
agent = create_agent(
    model=llm,
    response_format=ProviderStrategy(MySchema)  # Will fail or disable thinking
)

# ❌ Cannot use guided_json with tools in the same call
agent = create_agent(
    model=llm,
    tools=[my_tool],  # Tools will be removed when structured output is used
    response_format=ProviderStrategy(MySchema)
)

# ✅ Use structured output without thinking/tools
llm = ChatQwenVllm(enable_thinking=False)
agent = create_agent(
    model=llm,
    tools=[],  # No tools
    response_format=ProviderStrategy(MySchema)
)

# ✅ Or use with_structured_output for more flexibility
structured_llm = llm.with_structured_output(MySchema, method="json_schema")
result = structured_llm.invoke("Your query")
```

### Provider Strategy Requirement

Always use `ProviderStrategy` explicitly with `create_agent`:

```python
# ✅ Correct - Explicit ProviderStrategy
from langchain.agents.structured_output import ProviderStrategy
response_format=ProviderStrategy(MySchema)

# ❌ Not recommended - May fall back to ToolStrategy
response_format=MySchema
```

## 📊 Advanced Examples

### Multi-step Analysis with Multiple Tools

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def analyze_data(data_description: str) -> str:
    """Analyze data and provide insights."""
    return "Statistical analysis completed with key findings..."

@tool
def create_chart(data: str, chart_type: str) -> str:
    """Create a visualization chart."""
    return f"Created {chart_type} chart from data"

# Agent with multiple analysis tools
agent = create_agent(
    model=llm,
    tools=[analyze_data, create_chart],
    system_prompt="You are a data analyst. Use tools to analyze data and create visualizations."
)

result = agent.invoke({
    "messages": [
        HumanMessage(content="Analyze sales trends and create a bar chart")
    ]
})
```

### Async Operations with Agents

```python
import asyncio
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def quick_search(query: str) -> str:
    """Perform a quick search."""
    return f"Search results for: {query}"

async def process_requests():
    """Process multiple agent requests asynchronously."""
    
    agent = create_agent(
        model=llm,
        tools=[quick_search],
        system_prompt="You are a helpful research assistant."
    )
    
    # Process multiple queries concurrently
    queries = [
        "Analyze renewable energy trends",
        "Explain machine learning basics",
        "Describe quantum computing applications"
    ]
    
    for query in queries:
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=query)]
        })
        print(f"Response: {result['messages'][-1].content[:100]}...")

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
| LangChain 1.0 Agents | ❌ | ✅ Full Support |
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
│   └── integration_tests/      # LangChain 1.0 agent integration tests
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

# LangChain 1.0 agent tests
pytest tests/integration_tests/test_chat_models_vllm_langchain_agent.py
```

## 🚀 Getting Started

1. **Start VLLM server**:
   ```bash
   vllm serve Qwen/Qwen3-32B --port 8000
   ```

2. **Install dependencies**:
   ```bash
   pip install langchain-qwq
   # Note: Requires LangChain 1.0+
   ```

3. **Run examples**:
   ```bash
   python docs/tool_calling_example.py
   python docs/vllm_structured_output_example.py
   ```

## 🤝 Contributing

This project focuses on VLLM optimization and LangChain 1.0 integration. Contributions are welcome for:

- VLLM performance optimizations
- LangChain 1.0 agent workflow examples
- Advanced reasoning + structured output use cases
- Documentation improvements

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Original Project**: [@yigit353/langchain-qwq](https://github.com/yigit353/langchain-qwq)
- **Development Tool**: This project was entirely developed using [Cursor](https://cursor.so/) AI-powered code editor
- **Framework**: Built on [LangChain 1.0](https://github.com/langchain-ai/langchain) and [VLLM](https://github.com/vllm-project/vllm)

---

**✨ Created with [Cursor](https://cursor.so/) - The AI-powered code editor**