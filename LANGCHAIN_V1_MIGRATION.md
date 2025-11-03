# LangChain 1.0 Migration Summary

本文档记录了 `langchain-qwq-vllm` 从 LangChain 0.3 迁移到 LangChain 1.0 的完整过程和实现细节。

## 📋 迁移概览

### 版本升级
- **LangChain Core**: 0.3.x → 1.0.x
- **LangChain**: 0.3.x → 1.0.x  
- **LangGraph**: 0.2.x → 1.0.x
- **项目版本**: 0.0.7 → 1.0.0

### 关键变更
1. ✅ 更新所有 LangChain 依赖到 1.0.x 版本
2. ✅ 适配 LangChain 1.0 的 `create_agent()` API
3. ✅ 移除 DeepAgent 依赖，使用标准 LangChain agents
4. ✅ 实现 VLLM 原生结构化输出支持（Provider Strategy）
5. ✅ 更新测试用例适配新的 agent 框架
6. ✅ 更新文档和示例代码

## 🚀 新功能实现

### 1. 结构化输出支持

#### 实现方式
通过 VLLM 的 `guided_json` 参数实现原生结构化输出，支持 LangChain 1.0 的 Provider Strategy 模式。

#### 核心实现
在 `ChatQwenVllm` 类中添加：

```python
def _supports_structured_output(self) -> bool:
    """指示此模型支持通过 guided_json 的原生结构化输出。"""
    return True

def _get_request_payload(self, input_, *, stop=None, **kwargs) -> dict:
    """重写以处理结构化输出的 guided_json。"""
    payload = super()._get_request_payload(input_, stop=stop, **kwargs)
    
    # 将 OpenAI 的 response_format 转换为 VLLM 的 guided_json
    if 'response_format' in payload:
        response_format = payload['response_format']
        if isinstance(response_format, dict):
            if response_format.get('type') == 'json_schema':
                json_schema_data = response_format.get('json_schema', {})
                schema = json_schema_data.get('schema')
                
                if schema:
                    # 添加 guided_json 到 extra_body
                    payload['extra_body'] = {'guided_json': schema}
                    
                    # 移除 response_format（VLLM 不支持）
                    del payload['response_format']
                    
                    # VLLM 不支持 tools + guided_json 同时使用
                    if 'tools' in payload:
                        del payload['tools']
                    if 'parallel_tool_calls' in payload:
                        del payload['parallel_tool_calls']
    
    return payload
```

#### 使用方式

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy

class ContactInfo(BaseModel):
    """联系信息"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱")
    phone: str = Field(description="电话")

# 创建带结构化输出的 agent
agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="提取联系信息",
    response_format=ProviderStrategy(ContactInfo)  # 必须显式使用 ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "张三，zhangsan@example.com，138-1234-5678"}]
})

contact = result["structured_response"]
```

#### 重要说明
- **必须使用 `ProviderStrategy`**: LangChain 的 `_supports_provider_strategy()` 函数只识别特定模型名称（gpt-5, grok等），不会自动为 ChatQwenVllm 选择 Provider Strategy
- **限制**: VLLM 不支持 `guided_json` 与 `tools` 或 `enable_thinking` 同时使用

### 2. Agent 集成

#### 从 DeepAgent 迁移到 create_agent

**旧方式（LangChain 0.3）：**
```python
from deepagents import DeepAgent

agent = DeepAgent(
    llm=llm,
    tools=[calculator, search],
    system_message="You are a helpful assistant"
)
```

**新方式（LangChain 1.0）：**
```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[calculator, search],
    system_prompt="You are a helpful assistant"
)

# create_agent 返回 CompiledStateGraph
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Calculate 2+2"}
    ]
})
```

#### Agent 执行流程
1. `create_agent()` 返回 `CompiledStateGraph` 对象
2. 使用 `.invoke()` 方法执行，传入包含 `messages` 的字典
3. 结果包含 `messages` 键（对话历史）和可选的 `structured_response` 键

## 📦 依赖变更

### pyproject.toml 更新

```toml
[tool.poetry.dependencies]
python = ">=3.9,<4.0"
langchain-core = "^1.0.0"
openai = "^1.70.0"
langchain-openai = "^1.0.0"
python-dotenv = "^1.1.0"
json-repair = "^0.40.0"

[tool.poetry.group.test.dependencies]
pytest = "^7.4.3"
pytest-asyncio = "^0.23.2"
pytest-socket = "^0.7.0"
pytest-watcher = "^0.3.4"
langchain-tests = "^1.0.0"
python-dotenv = "^1.1.0"

[tool.poetry.group.dev.dependencies]
langchain = "^1.0.0"
langgraph = "^1.0.0"
```

## 🧪 测试用例更新

### 删除的测试
- ❌ `test_chat_models.py` - 旧的通用测试
- ❌ `test_chat_models_vllm.py` - LangChain 标准测试套件
- ❌ `test_chat_models_with_deepagents.py` - DeepAgent 集成测试
- ❌ `test_compile.py` - 编译测试占位符
- ❌ `test_deepagents_setup.md` - DeepAgent 设置文档

### 新增的测试

#### 1. `test_chat_models_vllm_langchain_agent.py`
LangChain 1.x agent 集成测试：
- ✅ 基本 agent 创建
- ✅ 计算器工具执行
- ✅ 多工具 agent
- ✅ 启用思考模式的 agent
- ✅ 错误处理
- ✅ 流式 agent 执行
- ✅ enable_thinking 兼容性
- ✅ 工具绑定兼容性
- ✅ LangChain 版本验证
- ✅ create_agent 可用性检查

#### 2. `test_structured_output_with_agent.py`
结构化输出测试：
- ✅ 简单 Pydantic 模型提取
- ✅ 复杂嵌套结构
- ✅ 列表和数组
- ✅ 可选字段
- ✅ 枚举和字面量
- ✅ 字段验证
- ✅ 提取准确性
- ✅ 向后兼容性（with_structured_output）

### 测试结果
```bash
$ pytest tests/integration_tests/ -v
=================== 18 passed, 1 warning in 237.94s ====================
```

## 📚 文档更新

### README.md
- ✅ 更新特性说明，添加结构化输出支持
- ✅ 添加 LangChain 1.0 agent 集成示例
- ✅ 添加结构化输出使用示例
- ✅ 更新依赖和安装说明

### 新增示例文件
- ✅ `examples/structured_output_example.py` - 完整的结构化输出示例
  - 简单联系信息提取
  - 嵌套结构（Person with Address）
  - 复杂分析（ProductReview）
  - 可选字段（EventInfo）

### 新增文档
- ✅ `tests/integration_tests/README.md` - 集成测试说明
- ✅ `LANGCHAIN_V1_MIGRATION.md` - 本迁移文档

## 🔧 技术细节

### 关键技术挑战

#### 1. 结构化输出的 VLLM 适配
**问题**: LangChain 1.0 使用 OpenAI 的 `response_format` 参数，但 VLLM 使用 `guided_json`。

**解决方案**: 在 `_get_request_payload` 中拦截并转换：
- 检测 `response_format` 参数
- 提取 JSON schema
- 转换为 VLLM 的 `extra_body: {guided_json: schema}`
- 移除不兼容的参数（tools, enable_thinking）

#### 2. Provider Strategy 识别
**问题**: LangChain 的 `_supports_provider_strategy()` 不识别 ChatQwenVllm。

**解决方案**: 
- 实现 `_supports_structured_output()` 方法
- 在文档中明确说明需要显式使用 `ProviderStrategy`
- 在测试中统一使用 `ProviderStrategy(schema)` 而非直接传递 schema

#### 3. VLLM 参数冲突
**问题**: VLLM 不支持 `guided_json` 与其他参数同时使用。

**解决方案**: 
- 检测到 `guided_json` 时，清空 `extra_body` 其他内容
- 移除 `tools` 和 `parallel_tool_calls` 参数
- 在文档中明确说明这些限制

### 代码变更统计
- 修改文件: 5个核心文件
  - `langchain_qwq/chat_models_vllm.py`
  - `pyproject.toml`
  - `README.md`
  - `tests/integration_tests/*`
- 新增文件: 4个
  - `test_chat_models_vllm_langchain_agent.py`
  - `test_structured_output_with_agent.py`
  - `examples/structured_output_example.py`
  - `LANGCHAIN_V1_MIGRATION.md`
- 删除文件: 5个旧测试文件

## 🎯 验证清单

- [x] 所有依赖更新到 LangChain 1.x
- [x] Agent 使用 `create_agent()` 而非 DeepAgent
- [x] 结构化输出支持 Provider Strategy
- [x] 测试用例全部适配新框架
- [x] 文档和示例更新完整
- [x] 向后兼容性保持（with_structured_output）
- [x] 集成测试通过率 > 95%

## 📝 使用注意事项

### 1. 结构化输出限制
- **必须显式使用 `ProviderStrategy`**
- 不能与 `enable_thinking` 同时使用
- 不能与 `tools` 同时使用（在 agent 中）

### 2. Agent 创建
```python
# ✅ 正确
agent = create_agent(model=llm, tools=[...], system_prompt="...")

# ❌ 错误（旧方式）
agent = DeepAgent(llm=llm, tools=[...])
```

### 3. 结构化输出
```python
# ✅ 正确
agent = create_agent(
    model=llm,
    response_format=ProviderStrategy(MySchema)
)

# ⚠️ 不推荐（会使用 ToolStrategy）
agent = create_agent(
    model=llm,
    response_format=MySchema  # 没有显式指定 ProviderStrategy
)
```

## 🔄 后续改进建议

1. **自动识别支持**: 贡献 PR 到 LangChain，将 ChatQwenVllm 添加到 `_supports_provider_strategy()` 的识别列表

2. **参数智能处理**: 检测参数冲突时给出更友好的警告信息

3. **异步支持优化**: 进一步优化异步模式下的结构化输出

4. **文档完善**: 添加更多实际应用场景的示例

## 📞 联系与支持

- **问题反馈**: 请在 GitHub Issues 中提交
- **功能请求**: 欢迎在 GitHub Discussions 中讨论
- **贡献代码**: 欢迎提交 Pull Request

---

**迁移完成日期**: 2025-11-03  
**LangChain 版本**: 1.0.x  
**项目版本**: 1.0.0

