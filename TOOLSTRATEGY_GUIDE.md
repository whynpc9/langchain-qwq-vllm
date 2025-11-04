# ToolStrategy Guide for vLLM+Qwen3

## ✅ Required Configuration

For structured output with agents in vLLM+Qwen3 deployments, you **must** use:

```python
from langchain.agents.structured_output import ToolStrategy

llm = ChatQwenVllm(
    model="Qwen/Qwen3-32B",
    enable_thinking=True  # ✅ MUST be True
)

agent = create_agent(
    model=llm,
    tools=[...],  # Can have tools or be empty
    response_format=ToolStrategy(  # ✅ Must use ToolStrategy
        schema=YourSchema,
        tool_message_content="Task complete!"
    )
)
```

## Why This Configuration?

### 1. enable_thinking=True is Required

**Observation from Testing:**
- With `enable_thinking=False`: Agent fails with Internal Server Error
- With `enable_thinking=True`: Agent works correctly

**Reason:**
- Qwen3 models require thinking mode for proper tool-calling behavior with vLLM
- The reasoning process helps the model correctly sequence tool calls and structured output

### 2. ToolStrategy (Not ProviderStrategy)

**Why ToolStrategy?**
- vLLM cannot handle `guided_json` and `tools` parameters simultaneously
- ToolStrategy converts the structured output schema into a special tool
- This allows tools and structured output to coexist

**How It Works:**
```
User Query
    ↓
Agent calls regular tools (calculator, search, etc.)
    ↓
Agent calls "structured_output" tool (created by ToolStrategy)
    ↓
Returns structured data
```

## Complete Examples

### Example 1: Simple Data Extraction (No Tools)

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")

llm = ChatQwenVllm(
    model="Qwen/Qwen3-32B",
    enable_thinking=True
)

agent = create_agent(
    model=llm,
    tools=[],  # No tools needed
    system_prompt="Extract contact information from text.",
    response_format=ToolStrategy(
        schema=ContactInfo,
        tool_message_content="Extraction complete!"
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "John Doe, john@example.com, (555) 123-4567"}]
})

contact = result["structured_response"]
print(f"Name: {contact.name}, Email: {contact.email}")
```

### Example 2: Agent with Tools and Structured Output

```python
from langchain_core.tools import tool

@tool
def calculator(operation: str, a: float, b: float) -> str:
    """Perform math operations."""
    ops = {
        "add": lambda x, y: x + y,
        "multiply": lambda x, y: x * y,
    }
    return str(ops[operation](a, b))

class MathReport(BaseModel):
    problem: str = Field(description="The math problem")
    steps: list[str] = Field(description="Solution steps")
    final_answer: float = Field(description="Final answer")

llm = ChatQwenVllm(
    model="Qwen/Qwen3-32B",
    enable_thinking=True
)

agent = create_agent(
    model=llm,
    tools=[calculator],  # Agent can use this tool
    system_prompt="Solve math problems using the calculator tool.",
    response_format=ToolStrategy(
        schema=MathReport,
        tool_message_content="Problem solved!"
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Calculate (5 + 3) * 2"}]
})

report = result["structured_response"]
print(f"Problem: {report.problem}")
print(f"Steps: {report.steps}")
print(f"Answer: {report.final_answer}")
```

## Common Issues and Solutions

### Issue 1: "Internal Server Error"

**Symptom:**
```
openai.InternalServerError: Internal Server Error
```

**Solution:**
Ensure `enable_thinking=True`:
```python
llm = ChatQwenVllm(enable_thinking=True)  # ✅
```

### Issue 2: Agent Stops After First Response

**Symptom:**
- Only one request to vLLM
- `finish_reason="stop"` immediately

**Solution:**
This is usually caused by `enable_thinking=False`. Set it to `True`.

### Issue 3: Tools Not Being Called

**Symptom:**
- Agent returns response without using available tools

**Solution:**
1. Ensure tool descriptions are clear
2. Verify `enable_thinking=True`
3. Make system prompt explicit about using tools

### Issue 4: No structured_response in Result

**Symptom:**
```python
KeyError: 'structured_response'
```

**Solution:**
- Verify you're using `ToolStrategy`, not `ProviderStrategy`
- Check that agent completed successfully

## Testing Configuration

In your tests, always use:

```python
def setup_method(self):
    self.llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        temperature=0.1,
        max_tokens=2000,
        enable_thinking=True,  # ✅ Required
    )
```

## Summary

| Component | Value | Required |
|-----------|-------|----------|
| `enable_thinking` | `True` | ✅ Yes |
| Strategy | `ToolStrategy` | ✅ Yes |
| `tool_message_content` | Any string | ✅ Yes |
| `tools` parameter | List or empty | ✅ Optional |

**Golden Rule:** For any agent with structured output on vLLM+Qwen3:
```python
enable_thinking=True + ToolStrategy
```
