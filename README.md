# langchain-qwq

This package contains the LangChain integration with QwQ and Qwen3

## Installation

```bash
pip install -U langchain-qwq
```

OR if you want to install additional dependencies when you clone the repo:

```bash
pip install -U langchain-qwq[docs]
pip install -U langchain-qwq[test]
pip install -U langchain-qwq[codespell]
pip install -U langchain-qwq[lint]
pip install -U langchain-qwq[typing]
```
### Environment Variables

And you should configure credentials by setting the following environment variables:

* `DASHSCOPE_API_KEY`: Your DashScope API key for accessing QwQ or Qwen3 models
* `DASHSCOPE_API_BASE`: (Optional) API base URL, defaults to "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

## Chat Models (QwQ)

`ChatQwQ` class exposes chat models from QwQ. The integration works directly with a standard API key without requiring the Tongyi dependency.

```python
from langchain_qwq import ChatQwQ

llm = ChatQwQ()
llm.invoke("Sing a ballad of LangChain.")
```

### Advanced Usage

#### Streaming

```python
llm = ChatQwQ(model="qwq-plus")
for chunk in llm.stream("Write a short poem about AI"):
    print(chunk.content, end="")
```

#### Async Support

```python
llm = ChatQwQ(model="qwq-plus")
response = await llm.ainvoke("What is the capital of France?")
print(response.content)

# Streaming
async for chunk in llm.astream("Tell me about quantum computing"):
    print(chunk.content, end="")
```

#### Access to Reasoning Content

```python
response = llm.invoke("Explain how photosynthesis works")
content = response.content
reasoning = response.additional_kwargs.get("reasoning_content", "")
```

### Merge Reasoning Content to Content

```python
from langchain_qwq import ChatQwQ
from langchain_qwq.utils import convert_reasoning_to_content
model = ChatQwQ(
    model="qwq-plus"
)
for chunk in convert_reasoning_to_content(model.stream("hello")):
    print(chunk)
```

also support async

```python
from langchain_qwq.utils import aconvert_reasoning_to_content
async for chunk in aconvert_reasoning_to_content(model.astream("hello")):
    print(chunk)
```

and you can custom the think tag.
```python
from langchain_qwq.utils import convert_reasoning_to_content

async for chunk in aconvert_reasoning_to_content(
        model.astream("hello"), think_tag=("<Start>", "<End>")
    ):
        print(chunk)

```

#### Tool Calls

```python
from langchain_core.tools import tool

@tool
def get_current_weather(location: str, unit: str = "fahrenheit"):
    """Get the current weather in a given location"""
    return f"72 degrees and sunny in {location}"

llm = ChatQwQ(model="qwq-plus")
llm_with_tools = llm.bind_tools([get_current_weather])
response = llm_with_tools.invoke("What's the weather in San Francisco?")
```

also you can use `parallel_tool_calls` to enable parallel tool calls
```python
from langchain_qwq import ChatQwQ
from langchain_core.tools import tool


@tool
async def get_data_in_db(db_name: str) -> str:
    """get the data from the database"""
    return f"data from the database {db_name}"


model = ChatQwQ(model="qwq-plus").bind_tools([get_data_in_db])

print(
    model.invoke(
        "please get the data from the database user and animal",
        extra_body={"parallel_tool_calls": True},
    )
)

```

## Chat Models (Qwen3)
You can call Qwen3 through ChatQwQ, but we more strongly recommend using ChatQwen to call Qwen3 as it provides better support.

there is some examples code

```python
from langchain_qwq import ChatQwen


model = ChatQwen(model="qwen3-32b")

for chunk in model.stream("hello!"):
    print(chunk)

```
 Qwen3 is hybrid reasoning, so you can use enable_thinking to turn off the thinking process.

```python
from langchain_qwq import ChatQwen
from dotenv import load_dotenv


model = ChatQwen(model="qwen3-32b", enable_thinking=False)

for chunk in model.stream("hello!"):
    print(chunk)

```

Qwen3 also supports thinking_budget to control the length of the thinking process.

```python
from langchain_qwq import ChatQwen
from dotenv import load_dotenv

model = ChatQwen(model="qwen3-32b", thinking_budget=100)

for chunk in model.stream("hello!"):
    print(chunk)


```

Qwen3's integration is through ChatQwen, which inherits from QwQ, so you can use any features supported by ChatQwQ such as async, etc.


