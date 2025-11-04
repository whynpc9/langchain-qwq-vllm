# 测试工具调用指南 (Testing Tool Calling Guide)

## 概述

本文档说明如何正确测试 `ChatQwenVllm` 与 LangChain 1.x Agent 的工具调用功能。

## 问题诊断

当使用 `create_agent` 时，虽然传入了 tools 参数，但如果只检查 agent 创建成功，并不能确保工具真的被调用。通过抓包发现，发送到 vllm 的请求中可能没有包含 tools 参数。

## 正确的测试方法

### 1. 添加工具调用追踪

在工具函数中添加计数器，确保工具被实际调用：

```python
from langchain_core.tools import tool

tool_call_count = {"count": 0}

@tool
def calculator(operation: str, a: float, b: float) -> str:
    """Perform basic math operations."""
    tool_call_count["count"] += 1
    print(f"[TOOL CALL] calculator: operation={operation}, a={a}, b={b}")
    
    # ... implementation
    return result

# 测试后验证
assert tool_call_count["count"] > 0, "Calculator tool was never called!"
```

### 2. 检查消息中的 tool_calls

验证返回的消息中包含 tool_calls：

```python
result = agent.invoke({
    "messages": [
        HumanMessage(content="Use the calculator to multiply 15 and 8")
    ]
})

# 检查是否有 tool_calls
has_tool_calls = False
for msg in result["messages"]:
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        has_tool_calls = True
        print(f"Tool calls found: {msg.tool_calls}")

assert has_tool_calls, "No tool_calls found in messages!"
```

### 3. 验证工具绑定

确保工具正确绑定到 LLM：

```python
from langchain_core.runnables.base import RunnableBinding

# 手动绑定工具
llm_with_tools = llm.bind_tools([test_tool])

# 验证绑定
assert isinstance(llm_with_tools, RunnableBinding)
assert 'tools' in llm_with_tools.kwargs

# 检查工具 schema
tools_schema = llm_with_tools.kwargs['tools']
assert len(tools_schema) > 0
assert 'function' in tools_schema[0]
print(f"Tool name: {tools_schema[0]['function']['name']}")
```

### 4. 明确指示 Agent 使用工具

在 prompt 中明确告诉 agent 使用工具：

```python
agent = create_agent(
    model=llm,
    tools=[calculator],
    system_prompt=(
        "You are a helpful math assistant. "
        "Use the calculator tool for all calculations."
    )
)

# 在用户消息中也明确要求使用工具
result = agent.invoke({
    "messages": [
        HumanMessage(content="Use the calculator to multiply 15 and 8")
    ]
})
```

## 完整测试示例

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_qwq.chat_models_vllm import ChatQwenVllm

# 初始化 LLM
llm = ChatQwenVllm(
    model="Qwen/Qwen3-32B",
    temperature=0.1,
    max_tokens=2000,
    enable_thinking=True,
)

# 定义带追踪的工具
tool_call_count = {"count": 0}

@tool
def calculator(operation: str, a: float, b: float) -> str:
    """Perform basic math operations."""
    tool_call_count["count"] += 1
    print(f"[TOOL CALL] calculator: {operation}({a}, {b})")
    
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: division by zero"
    }
    
    if operation in operations:
        result = operations[operation](a, b)
        return f"{a} {operation} {b} = {result}"
    return f"Error: unknown operation {operation}"

# 创建 agent
agent = create_agent(
    model=llm,
    tools=[calculator],
    system_prompt=(
        "You are a helpful math assistant. "
        "Use the calculator tool for all calculations."
    )
)

# 执行测试
result = agent.invoke({
    "messages": [
        HumanMessage(content="Use the calculator to multiply 15 and 8")
    ]
})

# 验证结果
assert "messages" in result
assert len(result["messages"]) > 0

# 验证工具被调用
assert tool_call_count["count"] > 0, "Calculator tool was never called!"
print(f"✓ Tool was called {tool_call_count['count']} time(s)")

# 验证消息中有 tool_calls
has_tool_calls = any(
    hasattr(msg, 'tool_calls') and msg.tool_calls
    for msg in result["messages"]
)
assert has_tool_calls, "No tool_calls found in agent messages!"
print("✓ Tool calls found in messages")

# 查看最终答案
final_message = result["messages"][-1]
print(f"Final answer: {final_message.content}")
```

## 常见问题

### Q: Agent 创建成功但工具从未被调用？

**A:** 可能的原因：
1. LLM 未正确绑定工具 - 使用 `bind_tools()` 验证
2. Prompt 不够明确 - 在 system_prompt 和用户消息中明确要求使用工具
3. VLLM 服务器未运行或配置错误
4. 模型本身不支持工具调用

### Q: 如何调试工具调用问题？

**A:** 
1. 在工具函数中添加 print 语句
2. 检查 result["messages"] 中的所有消息
3. 验证 tool_calls 是否存在
4. 使用抓包工具（如 tcpdump, Wireshark）查看发送到 VLLM 的实际请求

### Q: 结构化输出可以和工具调用同时使用吗？

**A:** 
不建议，VLLM 的限制：
- ❌ `guided_json` + `tools` 不能同时使用
- ❌ `guided_json` + `enable_thinking` 不能同时使用

如需同时使用，建议分两步：
1. 先用工具收集信息
2. 再用结构化输出格式化结果

## 运行测试

```bash
# 启动 VLLM 服务器
vllm serve Qwen/Qwen3-32B --port 8000

# 运行测试
pytest tests/integration_tests/test_chat_models_vllm_langchain_agent.py -v

# 只运行特定测试
pytest tests/integration_tests/test_chat_models_vllm_langchain_agent.py::TestChatQwenVllmWithLangChainAgent::test_agent_with_calculator -v
```

## 参考资料

- [LangChain 1.x Agent 文档](https://python.langchain.com/docs/modules/agents/)
- [VLLM Tool Calling 文档](https://docs.vllm.ai/en/latest/)
- [测试用例](../tests/integration_tests/test_chat_models_vllm_langchain_agent.py)


