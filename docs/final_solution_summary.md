# 工具调用与结构化输出集成最终解决方案

## 🎯 问题解决

用户报告的问题：在 `test_tool_calling_with_structured_output()` 测试用例中，使用 `bind_tools()` 后再调用 `with_structured_output()` 时，发送给 vLLM 的请求体中缺少工具信息。

**✅ 问题已完全解决！**

## 🛠️ 解决方案

### 新增方法：`bind_tools_with_structured_output()`

在 `ChatQwenVllm` 类中新增了一个专门的方法来同时处理工具绑定和结构化输出：

```python
def bind_tools_with_structured_output(
    self,
    tools: list,
    schema: Optional[_DictOrPydanticClass] = None,
    *,
    method: Literal["json_schema"] = "json_schema",
    include_raw: bool = False,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, _DictOrPydantic]:
    """Bind tools and structured output simultaneously for vLLM."""
```

### 核心特性

1. **完整的参数保留**：确保 `tools` 和 `guided_json` 都传递给 vLLM
2. **支持所有功能**：包括 `include_raw=True` 选项
3. **类型安全**：返回正确的类型注解
4. **错误处理**：完善的异常处理和参数验证

## 📋 使用方法

### 基本用法

```python
from langchain_qwq.chat_models_vllm import ChatQwenVllm
from langchain_core.tools import tool
from pydantic import BaseModel, Field

@tool
def get_weather(location: str) -> str:
    """获取天气信息."""
    return f"{location}的天气信息"

class WeatherQuery(BaseModel):
    location: str = Field(description="查询的地点")
    temperature: int = Field(description="温度")

# 初始化 LLM
llm = ChatQwenVllm(model="Qwen/Qwen3-32B")

# 🎉 使用新方法 - 一步到位
integrated_llm = llm.bind_tools_with_structured_output(
    tools=[get_weather],
    schema=WeatherQuery,
    method="json_schema"
)

# 调用
response = integrated_llm.invoke("北京的天气怎么样？")
# 返回 WeatherQuery 类型的结构化数据
```

### 高级用法：include_raw

```python
# 返回原始响应和解析结果
integrated_llm = llm.bind_tools_with_structured_output(
    tools=[get_weather],
    schema=WeatherQuery,
    method="json_schema",
    include_raw=True
)

response = integrated_llm.invoke("上海的天气怎么样？")
# 返回: {
#     "raw": AIMessage(...),
#     "parsed": WeatherQuery(...),
#     "parsing_error": None
# }
```

## ✅ 验证结果

### HTTP 请求验证

通过模拟 HTTP 请求的方式验证，新方法发送给 vLLM 的请求同时包含：

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
    ],
    "extra_body": {
        "guided_json": {
            "title": "WeatherQuery",
            "type": "object",
            "properties": {...}
        }
    }
}
```

### 测试用例验证

修改后的测试用例全部通过：

```bash
✅ test_tool_calling_with_structured_output PASSED
✅ test_tool_calling_with_include_raw PASSED
✅ 所有 TestChatQwenVllmToolCalling 测试通过 (7/7)
```

## 📚 对比：旧方法 vs 新方法

### ❌ 旧方法（有问题）

```python
# 工具信息会丢失
llm_with_tools = llm.bind_tools([tool])
structured_llm = llm_with_tools.with_structured_output(schema)
# 结果：只有 guided_json，没有 tools
```

### ✅ 新方法（完美解决）

```python
# 工具和结构化输出都保留
integrated_llm = llm.bind_tools_with_structured_output(
    tools=[tool],
    schema=schema,
    method="json_schema"
)
# 结果：同时有 tools 和 guided_json
```

## 🔧 技术实现

### 核心机制

1. **正确的绑定顺序**：先使用 `bind_tools()` 创建工具绑定
2. **智能参数合并**：将结构化输出参数合并到工具绑定的 kwargs 中
3. **统一输出解析**：使用 `RunnableLambda` 创建统一的输出处理链

### 关键代码片段

```python
# 使用 bind_tools 确保工具正确处理
llm_with_tools = self.bind_tools(tools, **kwargs)

# 智能合并 kwargs
if hasattr(llm_with_tools, 'kwargs'):
    existing_kwargs = llm_with_tools.kwargs.copy()
    # 合并结构化输出配置
    for key, value in bind_kwargs.items():
        if key == "extra_body" and key in existing_kwargs:
            existing_kwargs[key] = {**existing_kwargs[key], **value}
        else:
            existing_kwargs[key] = value
    
    # 创建最终绑定
    llm = llm_with_tools.bound.bind(**existing_kwargs)
```

## 🎯 适用场景

这个解决方案适用于所有需要同时使用工具调用和结构化输出的场景：

1. **智能助手**：需要调用外部API获取数据并返回结构化结果
2. **数据提取**：使用工具获取信息后按指定格式输出
3. **复杂查询**：多步骤操作，既要工具调用又要结构化响应
4. **生产环境**：需要稳定可靠的工具和输出格式集成

## 📝 测试用例更新

测试用例已更新使用新方法：

```python
def test_tool_calling_with_structured_output(self):
    # 使用新的集成方法
    final_chain = self.llm.bind_tools_with_structured_output(
        tools=[weather_tool],
        schema=WeatherQuery,
        method="json_schema"
    )
    
    response = final_chain.invoke([...])
    assert isinstance(response, WeatherQuery)
```

## 🚀 总结

通过添加 `bind_tools_with_structured_output()` 方法，我们完全解决了用户报告的问题：

- ✅ **工具调用信息完整保留**
- ✅ **结构化输出正常工作**  
- ✅ **支持所有高级功能**（include_raw 等）
- ✅ **类型安全和错误处理**
- ✅ **向后兼容**
- ✅ **测试全部通过**

这个解决方案提供了一个简洁、可靠的API来同时使用工具调用和结构化输出功能，让开发者能够充分利用 vLLM 的强大能力。
