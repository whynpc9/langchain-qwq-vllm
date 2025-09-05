# 工具调用与结构化输出集成解决方案

## 问题总结

在 `test_tool_calling_with_structured_output()` 测试用例中发现，当同时使用 `bind_tools()` 和 `with_structured_output()` 时，发送给 vLLM 的请求中缺少工具信息。

### 根本原因

LangChain 的 `RunnableBinding` 机制存在一个设计限制：
1. `llm.bind_tools([tool])` 返回一个 `RunnableBinding` 对象，包含 `kwargs={'tools': [...]}`
2. 调用 `binding.with_structured_output()` 时，LangChain 将调用委托给绑定的 `ChatQwenVllm` 对象
3. 但是委托调用中，`ChatQwenVllm.with_structured_output()` 方法无法访问 `RunnableBinding` 的 kwargs
4. 结果导致工具信息丢失，只保留了 `guided_json` 参数

### 技术分析

```python
# 问题流程
llm = ChatQwenVllm(...)
llm_with_tools = llm.bind_tools([tool])        # RunnableBinding(kwargs={'tools': [...]})
structured_llm = llm_with_tools.with_structured_output(schema)  # 工具信息丢失

# 实际执行路径：
# 1. RunnableBinding.with_structured_output() 被调用
# 2. LangChain 委托给 ChatQwenVllm.with_structured_output()
# 3. 但委托时没有传递 RunnableBinding 的 kwargs
# 4. ChatQwenVllm 只能访问自己的属性，无法获取工具信息
```

## 解决方案

### 方案 1: 辅助方法 (推荐)

添加一个便捷方法来同时绑定工具和结构化输出：

```python
def bind_tools_and_structured_output(
    self,
    tools: List[Tool],
    schema: Union[Dict, BaseModel],
    method: str = "json_schema",
    include_raw: bool = False,
    **kwargs
) -> Runnable:
    """同时绑定工具和结构化输出."""
    # 首先绑定工具
    llm_with_tools = self.bind_tools(tools, **kwargs)
    
    # 然后手动构建包含工具的 bind_kwargs
    bind_kwargs = {'tools': tools}  # 手动保留工具信息
    
    # 添加结构化输出配置
    if schema:
        from langchain_core.utils.function_calling import convert_to_json_schema
        schema_dict = convert_to_json_schema(schema)
        extra_body = {'guided_json': schema_dict}
        bind_kwargs['extra_body'] = extra_body
        bind_kwargs['ls_structured_output_format'] = {
            'kwargs': {'method': method},
            'schema': schema,
        }
    
    # 创建最终的绑定
    return self.bind(**bind_kwargs)
```

### 方案 2: 修复现有实现

在 `with_structured_output()` 方法中检测绑定上下文：

```python
# 当前实现 (已添加到代码中)
bind_kwargs = {}

if hasattr(self, 'kwargs') and self.kwargs:
    # 直接在 RunnableBinding 上调用
    bind_kwargs.update(self.kwargs)
elif '_bound_kwargs' in kwargs:
    # 通过显式传递绑定 kwargs
    bind_kwargs.update(kwargs.pop('_bound_kwargs'))

# 添加 guided_json 和其他参数...
```

### 方案 3: 显式传递绑定信息

用户可以通过参数显式传递绑定信息：

```python
# 使用方式
llm_with_tools = llm.bind_tools([tool])
structured_llm = llm_with_tools.with_structured_output(
    schema=WeatherQuery,
    method="json_schema",
    _bound_kwargs=llm_with_tools.kwargs  # 显式传递
)
```

## 当前状态

### 测试结果

测试 `test_tool_calling_with_structured_output` 通过，但实际的 HTTP 请求分析显示：

```json
{
  "model": "Qwen/Qwen3-32B",
  "messages": [...],
  "extra_body": {
    "guided_json": {
      "title": "WeatherQuery",
      "properties": {...}
    }
  }
  // ❌ 缺少 "tools" 字段
}
```

### 为什么测试通过？

测试通过可能是因为：
1. 模型通过系统提示推断需要查询天气
2. 直接生成符合 schema 的结构化数据
3. 没有实际使用工具调用机制

但这不是期望的行为，用户希望同时支持工具调用和结构化输出。

## 推荐解决方案

### 短期解决方案：使用反向顺序

```python
# 推荐的调用顺序
structured_llm = llm.with_structured_output(schema=WeatherQuery, method="json_schema")
final_llm = structured_llm.bind_tools([weather_tool])

# 这样可以确保工具信息不会丢失
```

### 长期解决方案：添加辅助方法

在 `ChatQwenVllm` 中添加：

```python
def bind_tools_with_structured_output(
    self,
    tools: List[Tool],
    schema: Union[Dict, BaseModel],
    **kwargs
) -> Runnable:
    """同时绑定工具和结构化输出，确保两者都正确传递给 vLLM."""
    # 实现细节...
```

## 验证方法

使用以下脚本验证修复效果：

```python
# 检查实际发送的 HTTP 请求参数
def verify_integration():
    with patch('openai.resources.chat.completions.Completions.create') as mock:
        llm_with_tools = llm.bind_tools([tool])
        structured_llm = llm_with_tools.with_structured_output(schema)
        structured_llm.invoke("测试消息")
        
        # 检查 mock.call_args 中的 tools 和 extra_body
        call_kwargs = mock.call_args.kwargs
        has_tools = 'tools' in call_kwargs
        has_guided_json = 'extra_body' in call_kwargs and 'guided_json' in call_kwargs['extra_body']
        
        return has_tools and has_guided_json
```

## 结论

当前的实现无法完全解决 `RunnableBinding` 委托调用的问题，因为这是 LangChain 框架的设计限制。

建议用户：
1. 使用反向绑定顺序作为临时解决方案
2. 等待完整的辅助方法实现
3. 或使用显式传递 `_bound_kwargs` 的方式

这个问题的根本解决需要在 LangChain 框架级别进行修改，或提供专门的集成方法。
