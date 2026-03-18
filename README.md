# langchain-qwq-vllm

`langchain-qwq-vllm` is a Qwen chat model integration for vLLM with explicit support
for LangChain 1.x `create_agent()`.

This fork focuses on three practical problems:

- legacy vs. modern vLLM payload differences
- Qwen3 vs. Qwen3.5 behavior differences
- structured output and tool-calling compatibility under LangChain 1.x

## Requirements

- Python `>=3.10`
- LangChain `>=1.0.0`
- `langchain-openai >=1.0.0`
- `openai >=1.109.1`

Python 3.9 is not supported for the LangChain 1.x `create_agent()` path.

## Installation

```bash
git clone https://github.com/whynpc9/langchain-qwq-vllm.git
cd langchain-qwq-vllm
pip install -e .
```

Set a vLLM endpoint:

```bash
export VLLM_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy-key-for-vllm"
```

## Core model

```python
from langchain_qwq import ChatQwenVllm

llm = ChatQwenVllm(
    model="Qwen/Qwen3-32B",
    base_url="http://localhost:8000/v1",
    vllm_mode="legacy",   # or "modern"
    enable_thinking=True,
)
```

### `vllm_mode`

- `legacy`
  - keeps the historical behavior
  - converts provider structured output to `guided_json`
- `modern`
  - preserves native `response_format`
  - locally disables thinking for native structured-output calls
  - clamps tool/agent/structured-output requests to `4096` output tokens

## `create_agent()` examples

### Tools only

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def calculator(operation: str, a: float, b: float) -> str:
    """Perform basic math operations."""
    if operation == "multiply":
        return str(a * b)
    if operation == "add":
        return str(a + b)
    return "unsupported"

llm = ChatQwenVllm(
    model="Qwen/Qwen3-32B",
    base_url="http://172.16.4.232:8000/v1",
    vllm_mode="legacy",
    enable_thinking=True,
)

agent = create_agent(
    model=llm,
    tools=[calculator],
    system_prompt="Use calculator for all calculations.",
)
```

### ProviderStrategy

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy

class ContactInfo(BaseModel):
    name: str = Field(description="Person name")
    email: str = Field(description="Email address")

llm = ChatQwenVllm(
    model="Qwen/Qwen3.5-27B",
    base_url="http://172.16.4.232:8302/v1",
    vllm_mode="modern",
    enable_thinking=True,
)

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="Extract contact info.",
    response_format=ProviderStrategy(ContactInfo),
)
```

### ToolStrategy with tools

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class CalcResult(BaseModel):
    problem: str = Field(description="Original problem")
    final_answer: float = Field(description="Final answer")

agent = create_agent(
    model=llm,
    tools=[calculator],
    system_prompt="Use calculator, then return structured output.",
    response_format=ToolStrategy(CalcResult, tool_message_content="done"),
)
```

## Known limitations

See [docs/CREATE_AGENT_LIMITATIONS.md](docs/CREATE_AGENT_LIMITATIONS.md).

As of 2026-03-18:

- all three validated endpoints support `create_agent()` for:
  - tools only
  - `ProviderStrategy`
  - `ToolStrategy + tools + enable_thinking=True`
- `Qwen3` endpoints still have limitations when `enable_thinking=False`:
  - `ToolStrategy + tools + enable_thinking=False` can return `500`
  - tools-only agents may return an empty final answer even when the call itself does
    not raise

## Tests

### Stable unit tests

```bash
python3 -m pytest tests/unit_tests/test_chat_models_vllm.py -q
```

### Raw vLLM endpoint probes

```bash
python3 tests/scripts/validate_vllm_qwen_endpoints.py
```

### LangChain 1.x `create_agent()` endpoint probes

Use Python 3.10+ with a LangChain 1.x environment. One working example is:

```bash
PYTHONPATH=/tmp/langchain1py311:. python3.11 tests/scripts/validate_create_agent_endpoints.py
```

## Repository notes

- `tests/unit_tests/test_chat_models_vllm.py` contains the stable payload and parsing
  coverage
- `tests/scripts/validate_vllm_qwen_endpoints.py` validates raw model behavior
- `tests/scripts/validate_create_agent_endpoints.py` validates LangChain 1.x
  agent behavior against live endpoints
