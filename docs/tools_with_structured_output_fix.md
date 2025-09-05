# 工具调用与结构化输出集成问题修复

## 问题描述

在 `test_tool_calling_with_structured_output()` 测试用例中发现，当同时使用 `bind_tools()` 和 `with_structured_output()` 时，发送给 vLLM 的请求中没有携带工具信息。

### 问题现象
```python
# 这种使用方式会导致工具信息丢失
llm_with_tools = self.llm.bind_tools([weather_tool])
structured_llm = llm_with_tools.with_structured_output(schema=WeatherQuery, method="json_schema")
# 结果：structured_llm 发送的请求中不包含 tools 参数
```

### 根本原因

`with_structured_output()` 方法的原始实现中，调用 `self.bind()` 创建新实例时没有保留原有的绑定参数（如工具信息）：

```python
# 原始问题代码
llm = self.bind(
    extra_body=extra_body,
    ls_structured_output_format={...},
)
# 问题：只传递了新参数，丢失了原有的 kwargs（包含工具信息）
```

## 问题分析

### LangChain 的绑定机制

1. **bind_tools() 的结果**：
   ```python
   llm_with_tools = llm.bind_tools([tool])
   # 返回：RunnableBinding(bound=ChatQwenVllm, kwargs={'tools': [...]})
   ```

2. **kwargs 结构**：
   ```python
   llm_with_tools.kwargs = {
       'tools': [
           {
               'type': 'function',
               'function': {
                   'name': 'tool_name',
                   'description': '...',
                   'parameters': {...}
               }
           }
       ]
   }
   ```

3. **with_structured_output() 的问题**：
   - 创建新的绑定时没有传递现有的 `kwargs`
   - 导致工具信息在新实例中丢失

## 修复方案

### 1. 保留现有 kwargs

修改 `with_structured_output()` 方法，确保在创建新绑定时保留现有参数：

```python
# 修复后的代码
bind_kwargs = {}
if hasattr(self, 'kwargs') and self.kwargs:
    bind_kwargs.update(self.kwargs)  # 保留现有参数（包括工具）

# 处理 extra_body 合并
if 'extra_body' in bind_kwargs:
    existing_extra_body = bind_kwargs.get('extra_body', {}) or {}
    extra_body = {**existing_extra_body, **extra_body}

bind_kwargs.update({
    "extra_body": extra_body,
    "ls_structured_output_format": {...},
})

llm = self.bind(**bind_kwargs)
```

### 2. 智能合并机制

- **保留现有参数**：从 `self.kwargs` 复制所有现有绑定参数
- **智能合并 extra_body**：合并现有和新的 extra_body，确保 guided_json 优先
- **参数优先级**：新的结构化输出参数覆盖同名的现有参数

## 修复验证

### 1. 参数保留验证
```python
llm_with_tools = llm.bind_tools([tool])
structured_llm = llm_with_tools.with_structured_output(schema, method="json_schema")

# 验证：底层绑定保留了工具信息
assert 'tools' in structured_llm.kwargs  # ✅ 现在能通过
assert 'guided_json' in structured_llm.kwargs['extra_body']  # ✅ 也有结构化输出
```

### 2. 功能完整性验证
- ✅ 工具调用正常生成
- ✅ 结构化输出正常工作
- ✅ guided_json 参数正确传递
- ✅ 两种功能无冲突地协同工作

## 实际效果

### 修复前
```json
// 发送给 vLLM 的请求
{
    "model": "Qwen/Qwen3-32B",
    "messages": [...],
    "guided_json": {...}
    // ❌ 缺少 "tools" 字段
}
```

### 修复后
```json
// 发送给 vLLM 的请求
{
    "model": "Qwen/Qwen3-32B", 
    "messages": [...],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "...",
                "parameters": {...}
            }
        }
    ],
    "guided_json": {
        "type": "object",
        "properties": {...}
    }
}
```

## 技术细节

### 1. RunnableBinding 结构
```python
RunnableBinding(
    bound=ChatQwenVllm(...),           # 原始 LLM 实例
    kwargs={'tools': [...], ...},      # 绑定参数
    config={},                         # 配置
    config_factories=[]                # 配置工厂
)
```

### 2. 参数合并策略
```python
# 优先级：新参数 > 现有参数
bind_kwargs = self.kwargs.copy()      # 复制现有参数
bind_kwargs.update(new_params)        # 新参数覆盖
```

### 3. extra_body 特殊处理
```python
# 对象级合并，而非覆盖
existing_extra_body = kwargs.get('extra_body', {})
new_extra_body = {..., 'guided_json': schema}
final_extra_body = {**existing_extra_body, **new_extra_body}
```

## 最佳实践

### 1. 参数绑定顺序
推荐的使用顺序（任一顺序都能正常工作）：
```python
# 方式 1：先绑定工具，再添加结构化输出
llm_with_tools = llm.bind_tools([tool])
structured_llm = llm_with_tools.with_structured_output(schema)

# 方式 2：先添加结构化输出，再绑定工具
structured_llm = llm.with_structured_output(schema)
final_llm = structured_llm.bind_tools([tool])
```

### 2. 参数冲突处理
- **extra_body 冲突**：自动合并，新参数优先
- **其他参数冲突**：新参数覆盖现有参数
- **工具重复绑定**：后绑定的工具列表覆盖前面的

### 3. 验证建议
```python
# 验证工具和结构化输出都正确配置
assert hasattr(final_llm, 'kwargs')
assert 'tools' in final_llm.kwargs
assert 'extra_body' in final_llm.kwargs
assert 'guided_json' in final_llm.kwargs['extra_body']
```

## 测试结果

修复后，`test_tool_calling_with_structured_output` 测试通过：

```bash
test_tool_calling_with_structured_output PASSED [100%] (10.44s)
```

该测试验证了：
- 工具调用生成正常
- 结构化输出格式正确
- 两种功能无冲突地协同工作
- 最终响应包含正确的结构化数据

## 总结

这个修复确保了 `ChatQwenVllm` 的 `with_structured_output()` 方法与 `bind_tools()` 可以无缝协作，解决了参数丢失的问题，让用户能够同时享受工具调用和结构化输出两种强大功能。

修复的核心思想是：**在创建新绑定时，始终保留现有的绑定参数，确保功能的累积性而非替换性**。
