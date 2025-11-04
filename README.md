# langchain-qwq-vllm

A VLLM-optimized integration package for Qwen3 models with **LangChain 1.0**, forked from [@yigit353/langchain-qwq](https://github.com/yigit353/langchain-qwq). This implementation is specifically designed for VLLM deployments of Qwen3 models and uses the modern LangChain 1.x agent framework.

> **📌 Version Notice**: This is the **LangChain 1.0** version (v1.0.0+). For the legacy LangChain 0.3 version, please check out the [`classic`](https://github.com/yourusername/langchain-qwq-vllm/tree/classic) branch.

## ✨ Key Advantages

This fork provides essential enhancements for VLLM environments:

- **🔧 VLLM Native Support**: Optimized for VLLM backend with Qwen3 models
- **🧠 Thinking + Structured Output**: Unique capability to combine reasoning mode with structured output via ToolStrategy
- **🤖 LangChain 1.0 Agents**: Full compatibility with modern LangChain 1.x agent framework using `create_agent()`
- **🛠️ Tools + Structured Output**: ToolStrategy enables tools and structured output to work together
- **⚡ Enhanced Performance**: Optimized for local/self-hosted VLLM deployments

### 🎯 Quick Start for Structured Output

For **vLLM+Qwen3**, always use this configuration:

```python
llm = ChatQwenVllm(enable_thinking=True)  # ✅ Must be True
agent = create_agent(
    model=llm,
    tools=[...],
    response_format=ToolStrategy(schema=YourSchema)  # ✅ Use ToolStrategy
)
```

### What's New in v1.0.0

- ✅ Migrated to LangChain 1.0+ (langchain-core, langchain, langgraph)
- ✅ Replaced DeepAgent with standard `create_agent()` API
- ✅ Implemented ToolStrategy for structured output with Qwen3+vLLM
- ✅ Enable_thinking=True configuration for optimal agent performance
- ✅ Added comprehensive integration tests for LangChain 1.x
- ✅ Improved documentation and examples

For migration guide from 0.x to 1.0, see [LANGCHAIN_V1_MIGRATION.md](LANGCHAIN_V1_MIGRATION.md).

## 🚀 Features

- **LangChain 1.0 Ready**: Fully compatible with LangChain 1.x, LangGraph 1.x
- **Streaming Support**: Real-time synchronous and asynchronous streaming
- **Reasoning Access**: Direct access to model's internal thinking process with `enable_thinking`
- **Structured Output**: ToolStrategy for structured output with Qwen3+vLLM agents
- **Tool Calling**: Function calling with parallel execution support
- **Tools + Structured Output**: Unique combination via ToolStrategy (requires `enable_thinking=True`)
- **Modern Agent Framework**: Full support for LangChain 1.0's `create_agent()` pattern
- **Comprehensive Tests**: 12+ integration tests covering all features


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

**🌟 New in v1.0: ToolStrategy for structured output with Qwen3+vLLM!**

For vLLM+Qwen3 deployments, you must use `ToolStrategy` with `enable_thinking=True`:

```python
from pydantic import BaseModel, Field
from typing import List
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class AnalysisResult(BaseModel):
    summary: str = Field(description="Brief analysis summary")
    key_points: List[str] = Field(description="Key findings")
    confidence: float = Field(description="Confidence score 0-1")

# Initialize LLM with thinking enabled (required for structured output)
llm = ChatQwenVllm(
    model="Qwen/Qwen3-32B",
    temperature=0.1,
    enable_thinking=True,  # ✅ Must be True for structured output with Qwen3
)

# Create agent with structured output using ToolStrategy
agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are an expert analyst.",
    response_format=ToolStrategy(  # Use ToolStrategy, not ProviderStrategy
        schema=AnalysisResult,
        tool_message_content="Analysis complete!"
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze the benefits of renewable energy"}]
})

analysis = result["structured_response"]
print(f"Summary: {analysis.summary}")
print(f"Key Points: {analysis.key_points}")
print(f"Confidence: {analysis.confidence}")
```

> **⚠️ Important**: For vLLM+Qwen3, always use `ToolStrategy` with `enable_thinking=True`. This combination is required for structured output to work correctly.

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

### Agent with Tools + Structured Output + Reasoning

The unique power of vLLM+Qwen3: combine reasoning, tool calling, and structured output:

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool

@tool
def calculator(operation: str, a: float, b: float) -> str:
    """Perform basic math operations."""
    ops = {"add": lambda x,y: x+y, "multiply": lambda x,y: x*y}
    return str(ops[operation](a, b))

class CalculationReport(BaseModel):
    problem: str = Field(description="Original math problem")
    steps: List[str] = Field(description="Solution steps")
    answer: float = Field(description="Final numerical answer")
    reasoning_summary: str = Field(description="Summary of reasoning process")

# Initialize with thinking enabled
llm = ChatQwenVllm(
    model="Qwen/Qwen3-32B",
    temperature=0.1,
    enable_thinking=True,  # ✅ Required for this combination
)

# Create agent with tools and structured output
agent = create_agent(
    model=llm,
    tools=[calculator],
    system_prompt="You are a math expert. Use tools and explain your reasoning.",
    response_format=ToolStrategy(
        schema=CalculationReport,
        tool_message_content="Calculation complete!"
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Calculate: (15 + 8) * 3"}]
})

report = result["structured_response"]
print(f"Problem: {report.problem}")
print(f"Steps: {report.steps}")
print(f"Answer: {report.answer}")
print(f"Reasoning: {report.reasoning_summary}")
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

### Agent with Tools and Structured Output

Combine agent tool calling with structured output using `ToolStrategy`:

```python
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool

@tool
def calculator(operation: str, a: float, b: float) -> str:
    """Perform basic math operations."""
    operations = {
        "add": lambda x, y: x + y,
        "multiply": lambda x, y: x * y,
        "subtract": lambda x, y: x - y,
    }
    return str(operations[operation](a, b))

class ResearchReport(BaseModel):
    topic: str = Field(description="Research topic")
    summary: str = Field(description="Executive summary")
    key_findings: list[str] = Field(description="Main findings")
    calculations: list[str] = Field(description="Any calculations performed")

# Initialize with thinking enabled
llm = ChatQwenVllm(
    model="Qwen/Qwen3-32B",
    temperature=0.1,
    enable_thinking=True,  # ✅ Required for Qwen3+vLLM
)

# Agent with tools that returns structured report
agent = create_agent(
    model=llm,
    tools=[calculator],  # Tools work with ToolStrategy
    system_prompt="You are a research assistant. Use tools when needed, then provide a structured report.",
    response_format=ToolStrategy(  # ToolStrategy allows tools + structured output
        schema=ResearchReport,
        tool_message_content="Research complete!"
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Calculate market growth: 2023 revenue $100M, 2024 $150M. What's the percentage increase?"}]
})

report = result["structured_response"]
print(f"Topic: {report.topic}")
print(f"Summary: {report.summary}")
print(f"Findings: {report.key_findings}")
print(f"Calculations: {report.calculations}")
```

> **💡 Key Point**: `ToolStrategy` converts your structured output schema into a special tool, allowing it to coexist with regular tools. This is essential for vLLM+Qwen3 which cannot handle `guided_json` and `tools` simultaneously.

## ⚠️ Important Requirements for vLLM+Qwen3

### Structured Output: Required Configuration

For structured output with agents in vLLM+Qwen3, you **must** use this specific combination:

```python
from langchain.agents.structured_output import ToolStrategy

# ✅ CORRECT: enable_thinking=True + ToolStrategy
llm = ChatQwenVllm(
    model="Qwen/Qwen3-32B",
    enable_thinking=True,  # ✅ Must be True for Qwen3
)

agent = create_agent(
    model=llm,
    tools=[...],  # Can have tools or empty list
    response_format=ToolStrategy(  # Must use ToolStrategy
        schema=MySchema,
        tool_message_content="Task complete!"
    )
)

# ❌ WRONG: enable_thinking=False causes failures
llm = ChatQwenVllm(enable_thinking=False)
agent = create_agent(
    model=llm,
    response_format=ToolStrategy(MySchema)  # Will fail with Qwen3
)

# ❌ WRONG: ProviderStrategy doesn't work with Qwen3
llm = ChatQwenVllm(enable_thinking=True)
agent = create_agent(
    model=llm,
    response_format=ProviderStrategy(MySchema)  # Don't use this
)
```

### Why This Configuration?

1. **enable_thinking=True**: Qwen3 models require thinking mode for proper tool-calling behavior
2. **ToolStrategy**: vLLM cannot handle `guided_json` (ProviderStrategy) with tools simultaneously
3. **Works with or without tools**: ToolStrategy converts the schema into a special tool that coexists with regular tools

### Strategy Comparison

| Configuration | Works? | Use Case |
|---------------|--------|----------|
| `enable_thinking=True` + `ToolStrategy` | ✅ | **Recommended for all agents with Qwen3** |
| `enable_thinking=False` + `ToolStrategy` | ❌ | Fails with Qwen3 |
| `enable_thinking=True` + `ProviderStrategy` | ❌ | Tools get removed |
| `enable_thinking=False` + `ProviderStrategy` | ❌ | Fails with Qwen3 |

### Direct LLM Usage (No Agent)

For non-agent scenarios, you can use `with_structured_output`:

```python
llm = ChatQwenVllm(
    model="Qwen/Qwen3-32B",
    enable_thinking=False  # Can be False for direct usage
)

structured_llm = llm.with_structured_output(
    schema=MySchema,
    method="json_schema"
)

result = structured_llm.invoke("Your query")
# Returns instance of MySchema
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

| Feature | Original (BaiLian) | Our VLLM Fork (Qwen3) |
|---------|-------------------|------------------------|
| Thinking Mode | ✅ | ✅ |
| Structured Output | ✅ | ✅ |
| **Thinking + Structured (ToolStrategy)** | ❌ | ✅ Required |
| **Tools + Structured Output** | ❌ | ✅ Via ToolStrategy |
| Tool Calling | ✅ | ✅ |
| LangChain 1.0 Agents | ❌ | ✅ Full Support |
| Local Deployment | ❌ | ✅ |
| Custom VLLM Optimizations | ❌ | ✅ |

**Key Difference**: Our implementation requires `enable_thinking=True` + `ToolStrategy` for structured output with agents, enabling unique combinations not possible with other implementations.

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

## 📚 Quick Reference Guide

### For Structured Output with Agents (Qwen3+vLLM)

**✅ Always Use This Pattern:**

```python
from langchain.agents.structured_output import ToolStrategy

llm = ChatQwenVllm(
    model="Qwen/Qwen3-32B",
    enable_thinking=True  # Must be True
)

agent = create_agent(
    model=llm,
    tools=[...],  # With or without tools
    response_format=ToolStrategy(
        schema=YourSchema,
        tool_message_content="Done!"
    )
)
```

### Common Patterns

| Task | Configuration | Example |
|------|---------------|---------|
| Data extraction | `enable_thinking=True` + `ToolStrategy` | Extract contact info |
| Agent with calculations | `enable_thinking=True` + `ToolStrategy` + `tools=[calculator]` | Math problem solver |
| Multi-tool workflow | `enable_thinking=True` + `ToolStrategy` + `tools=[...]` | Research assistant |
| Direct LLM call | `enable_thinking=False` + `with_structured_output()` | Simple query |

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "Internal Server Error" | Ensure `enable_thinking=True` |
| Tools not called | Check tool descriptions are clear |
| No structured response | Verify using `ToolStrategy`, not `ProviderStrategy` |
| finish_reason="stop" too early | Confirm `enable_thinking=True` |

## 🤝 Contributing

This project focuses on VLLM optimization and LangChain 1.0 integration. Contributions are welcome for:

- VLLM performance optimizations
- LangChain 1.0 agent workflow examples
- Advanced reasoning + structured output use cases with ToolStrategy
- Documentation improvements

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Original Project**: [@yigit353/langchain-qwq](https://github.com/yigit353/langchain-qwq)
- **Development Tool**: This project was entirely developed using [Cursor](https://cursor.so/) AI-powered code editor
- **Framework**: Built on [LangChain 1.0](https://github.com/langchain-ai/langchain) and [VLLM](https://github.com/vllm-project/vllm)

---

**✨ Created with [Cursor](https://cursor.so/) - The AI-powered code editor**