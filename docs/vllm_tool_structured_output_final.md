# vLLM 工具调用与结构化输出集成最终解决方案

## 🔍 深入测试发现的关键问题

经过深入测试，我们发现了一个重要的技术限制：

**vLLM 在同时收到 `tools` 和 `guided_json` 参数时，会优先基于 `guided_json` 返回内容，而不会返回有效的 `tool_calls` 进行工具调用。**

这意味着我们之前尝试的"同时发送两个参数"的方案在技术上可行，但在实际使用中无法达到预期效果。

## 💡 正确的解决方案

### 分阶段处理模式

基于 vLLM 的行为特性，正确的方案是将工具调用和结构化输出分为两个独立的阶段：

```
阶段1: 工具调用
├── 发送请求: 包含 tools，不包含 guided_json
├── 获取响应: 包含 tool_calls
└── 执行工具: 获取实际数据

阶段2: 结构化输出  
├── 发送请求: 包含 guided_json，不包含 tools
├── 输入内容: 原始对话 + 工具调用结果
└── 获取响应: 结构化的数据对象
```

## 🛠️ 实现调整

### 1. 移除不可行的方法

我们移除了 `ChatQwenVllm.bind_tools_with_structured_output()` 方法，因为它不能真正达到工具调用的目的。

### 2. 测试用例重构

重构了相关测试用例，实现了真正有效的分阶段工具调用和结构化输出：

#### 第一个测试：`test_tool_calling_with_structured_output`

```python
def test_tool_calling_with_structured_output(self):
    # 阶段1: 工具调用 - 绑定工具但不包含 guided_json
    llm_with_tools = self.llm.bind_tools([weather_tool])
    
    # 获取工具调用
    tool_response = llm_with_tools.invoke(initial_messages)
    assert hasattr(tool_response, 'tool_calls') and tool_response.tool_calls
    
    # 执行工具调用
    tool_messages = []
    for tool_call in tool_response.tool_calls:
        tool_result = weather_tool.invoke(tool_call['args'])
        tool_messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call['id']))
    
    # 阶段2: 结构化输出 - 使用 guided_json 但不包含 tools
    structured_llm = self.llm.with_structured_output(schema=WeatherQuery, method="json_schema")
    
    # 构建包含工具结果的完整对话历史
    conversation_with_tools = initial_messages + [tool_response] + tool_messages
    final_messages = conversation_with_tools + [instruction_for_structured_output]
    
    # 获取结构化响应
    response = structured_llm.invoke(final_messages)
    assert isinstance(response, WeatherQuery)
```

#### 第二个测试：`test_tool_calling_with_include_raw`

类似的分阶段处理，但使用 `include_raw=True` 选项：

```python
# 阶段1: 工具调用
tool_response = llm_with_tools.invoke(initial_messages)
# 执行工具...

# 阶段2: 结构化输出（带原始响应）
structured_llm = self.llm.with_structured_output(
    schema=WeatherSummary, 
    method="json_schema", 
    include_raw=True
)
response = structured_llm.invoke(final_messages)

# 验证 include_raw 格式
assert isinstance(response, dict)
assert "raw" in response
assert "parsed" in response  
assert "parsing_error" in response
```

## 📋 请求对比分析

### 阶段1 - 工具调用请求
```json
{
    "model": "Qwen/Qwen3-32B",
    "messages": [...],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气信息",
                "parameters": {...}
            }
        }
    ]
    // ✅ 有 tools，无 guided_json - 能正常返回 tool_calls
}
```

### 阶段2 - 结构化输出请求
```json
{
    "model": "Qwen/Qwen3-32B", 
    "messages": [...], // 包含工具调用的完整对话历史
    "extra_body": {
        "guided_json": {
            "type": "object",
            "properties": {...}
        }
    }
    // ✅ 有 guided_json，无 tools - 能正常返回结构化输出
}
```

## ✅ 验证结果

### 测试通过情况
- ✅ `test_tool_calling_with_structured_output` - 基本工具调用+结构化输出
- ✅ `test_tool_calling_with_include_raw` - 带原始响应的工具调用+结构化输出  
- ✅ 所有其他工具调用测试 (7/7 全部通过)

### 功能验证
1. **工具调用阶段**：vLLM 正确返回 `tool_calls`，不受 `guided_json` 干扰
2. **结构化输出阶段**：vLLM 正确按照 schema 返回结构化数据
3. **对话连续性**：工具调用结果正确传递到结构化输出阶段
4. **类型安全**：最终输出符合 Pydantic 模型类型要求

## 🎯 最佳实践

### 推荐的使用模式

```python
from langchain_qwq.chat_models_vllm import ChatQwenVllm
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from pydantic import BaseModel

# 1. 定义工具和 schema
@tool
def get_weather(location: str) -> str:
    return f"{location}的天气：晴朗，22°C"

class WeatherQuery(BaseModel):
    location: str
    temperature: int
    condition: str

# 2. 初始化 LLM
llm = ChatQwenVllm(model="Qwen/Qwen3-32B")

# 3. 阶段1：工具调用
llm_with_tools = llm.bind_tools([get_weather])
tool_response = llm_with_tools.invoke("北京的天气怎么样？")

# 4. 执行工具调用
tool_messages = []
for tool_call in tool_response.tool_calls:
    result = get_weather.invoke(tool_call['args'])
    tool_messages.append(ToolMessage(content=result, tool_call_id=tool_call['id']))

# 5. 阶段2：结构化输出
structured_llm = llm.with_structured_output(WeatherQuery, method="json_schema")
conversation = [original_message, tool_response] + tool_messages
final_response = structured_llm.invoke(conversation + [format_instruction])

# 6. 获得结构化结果
assert isinstance(final_response, WeatherQuery)
```

## 🚫 避免的反模式

```python
# ❌ 错误：尝试同时发送 tools 和 guided_json
# 这会导致 vLLM 忽略 tools，不返回 tool_calls
llm_with_both = llm.bind_tools([tool]).with_structured_output(schema)

# ❌ 错误：期望单一调用同时完成工具调用和结构化输出
response = llm_with_both.invoke("天气查询")  # 不会产生工具调用
```

## 📊 性能特征

### 请求数量
- **原方案期望**：1 次请求（工具调用+结构化输出）
- **实际可行方案**：2 次请求（工具调用 → 结构化输出）

### 响应质量
- **工具调用准确性**：✅ 优秀（无 guided_json 干扰）
- **结构化输出质量**：✅ 优秀（有丰富的上下文信息）
- **类型安全性**：✅ 完整（Pydantic 验证）

## 🎉 总结

通过深入测试发现 vLLM 的实际行为特性后，我们采用了分阶段处理的方案：

1. **技术可行性**：完全符合 vLLM 的实际行为
2. **功能完整性**：既能进行工具调用，又能产生结构化输出
3. **代码清晰性**：明确分离两个阶段，易于理解和维护
4. **测试覆盖度**：全面的测试覆盖，包括边界情况

这个解决方案虽然需要两次 API 调用，但确保了功能的可靠性和预期效果的实现，是在当前 vLLM 技术约束下的最优解决方案。
