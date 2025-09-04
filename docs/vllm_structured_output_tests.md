# ChatQwenVllm 结构化输出测试用例

本文档描述了为 `ChatQwenVllm` 的结构化输出功能新增的测试用例。

## 测试类: `TestChatQwenVllmStructuredOutput`

### 基础功能测试

#### 1. `test_structured_output_with_pydantic_model`
- **功能**: 测试使用 Pydantic 模型的结构化输出
- **验证**: 返回正确的 Pydantic 实例，所有字段类型正确
- **示例**: 从文本中提取人员信息 (姓名、年龄、职业、地点)

#### 2. `test_structured_output_with_dict_schema`
- **功能**: 测试使用字典 JSON Schema 的结构化输出
- **验证**: 返回正确的字典，包含所有必需字段
- **示例**: 从文本中提取书籍信息 (标题、作者、年份、类型)

#### 3. `test_structured_output_complex_nested_schema`
- **功能**: 测试复杂嵌套结构的处理
- **验证**: 正确处理嵌套的 Pydantic 模型
- **示例**: 公司信息包含地址和联系信息的嵌套结构

### 高级功能测试

#### 4. `test_structured_output_with_list_schema`
- **功能**: 测试列表/数组结构的处理
- **验证**: 正确解析列表中的多个项目
- **示例**: 任务列表，每个任务包含 ID、标题、优先级、完成状态

#### 5. `test_structured_output_with_include_raw`
- **功能**: 测试 `include_raw=True` 参数
- **验证**: 返回包含 raw、parsed、parsing_error 的字典
- **示例**: 天气信息提取，同时返回原始响应

#### 6. `test_structured_output_async`
- **功能**: 测试异步操作支持
- **验证**: 异步调用正常工作，返回正确类型
- **示例**: 书评分析的异步处理

### 错误处理测试

#### 7. `test_structured_output_error_unsupported_method`
- **功能**: 测试不支持的方法错误处理
- **验证**: 正确拒绝 `function_calling` 和 `json_mode` 方法
- **期望**: 抛出 `ValueError` 并包含相应错误信息

#### 8. `test_structured_output_error_missing_schema`
- **功能**: 测试缺少 schema 参数的错误处理
- **验证**: 当 schema=None 时正确抛出错误
- **期望**: 抛出 `ValueError` 提示必须提供 schema

#### 9. `test_structured_output_error_invalid_kwargs`
- **功能**: 测试无效参数的错误处理
- **验证**: 拒绝不支持的额外参数
- **期望**: 抛出 `ValueError` 列出不支持的参数

### 实际应用场景测试

#### 10. `test_structured_output_medical_coding_example`
- **功能**: 测试医疗编码分析的真实场景
- **验证**: 正确处理中文医疗文本和 ICD 编码匹配
- **示例**: 手术记录文本与 ICD 编码的匹配分析
- **结构**: 包含编码、名称、匹配标志、评分、规则等字段

### 技术验证测试

#### 11. `test_structured_output_schema_conversion`
- **功能**: 测试不同 schema 类型的转换
- **验证**: Pydantic 模型和字典 schema 都能正确处理
- **目的**: 确保 schema 转换逻辑正常工作

#### 12. `test_structured_output_extra_body_integration`
- **功能**: 测试 `guided_json` 与 `extra_body` 的集成
- **验证**: 结构化输出链能正确创建
- **目的**: 确保底层参数传递机制正常

## 运行测试

### 运行所有结构化输出测试
```bash
pytest tests/integration_tests/test_chat_models_vllm.py::TestChatQwenVllmStructuredOutput -v
```

### 运行特定测试
```bash
# 测试错误处理
pytest tests/integration_tests/test_chat_models_vllm.py::TestChatQwenVllmStructuredOutput::test_structured_output_error_unsupported_method -v

# 测试异步功能
pytest tests/integration_tests/test_chat_models_vllm.py::TestChatQwenVllmStructuredOutput::test_structured_output_async -v

# 测试医疗编码场景
pytest tests/integration_tests/test_chat_models_vllm.py::TestChatQwenVllmStructuredOutput::test_structured_output_medical_coding_example -v
```

## 注意事项

1. **vLLM 服务器要求**: 实际的 API 调用测试需要运行 vLLM 服务器
2. **仅支持 json_schema**: 测试确保只有 `json_schema` 方法被支持
3. **中文支持**: 包含中文医疗场景测试，验证多语言支持
4. **错误处理**: 全面覆盖各种错误场景，确保用户能获得清晰的错误信息

## 覆盖的功能点

- ✅ Pydantic 模型支持
- ✅ 字典 JSON Schema 支持
- ✅ 复杂嵌套结构
- ✅ 列表/数组处理
- ✅ include_raw 参数
- ✅ 异步操作
- ✅ 错误处理 (不支持的方法、缺少参数、无效参数)
- ✅ 真实应用场景 (医疗编码)
- ✅ vLLM 特定的 guided_json 集成
