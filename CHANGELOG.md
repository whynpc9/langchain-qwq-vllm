# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-03

### 🎉 Major Release - LangChain 1.0 Migration

This is a major version release migrating from LangChain 0.3 to LangChain 1.0.

### Added

#### Structured Output Support
- ✨ Native structured output support via VLLM's `guided_json`
- ✨ Provider Strategy implementation for LangChain 1.0 agents
- ✨ Support for Pydantic models, TypedDict, and JSON schema
- ✨ Comprehensive structured output examples in `examples/structured_output_example.py`
- ✨ Integration tests for structured output with agents

#### LangChain 1.0 Agent Integration
- ✨ Full support for LangChain 1.0's `create_agent()` API
- ✨ Compatibility with LangGraph's `CompiledStateGraph`
- ✨ Agent integration tests in `test_chat_models_vllm_langchain_agent.py`
- ✨ Support for agent execution with streaming
- ✨ Error handling in agent loops

#### Documentation
- 📚 Comprehensive migration guide (`LANGCHAIN_V1_MIGRATION.md`)
- 📚 Updated README with LangChain 1.0 examples
- 📚 Integration tests documentation (`tests/integration_tests/README.md`)
- 📚 Structured output examples and usage patterns

### Changed

#### Dependencies
- ⬆️ Upgraded `langchain-core` from 0.3.x to ^1.0.0
- ⬆️ Upgraded `langchain-openai` from 0.3.x to ^1.0.0
- ⬆️ Upgraded `langchain` from 0.3.x to ^1.0.0
- ⬆️ Upgraded `langgraph` from 0.2.x to ^1.0.0
- ⬆️ Upgraded `langchain-tests` to ^1.0.0

#### Core Implementation
- 🔧 Implemented `_supports_structured_output()` method
- 🔧 Added `_get_request_payload()` override for structured output handling
- 🔧 Improved `extra_body` parameter handling for VLLM compatibility
- 🔧 Enhanced error messages and validation

#### Test Suite
- ♻️ Refactored test suite for LangChain 1.0 compatibility
- ♻️ Replaced DeepAgent tests with standard LangChain agent tests
- ♻️ Added 19 new integration tests for LangChain 1.0 features
- ♻️ Improved test organization and documentation

### Removed

- 🗑️ Removed DeepAgent dependency and related tests
- 🗑️ Removed deprecated LangChain 0.3 test patterns
- 🗑️ Cleaned up legacy test files:
  - `test_chat_models.py`
  - `test_chat_models_vllm.py` (LangChain standard suite)
  - `test_chat_models_with_deepagents.py`
  - `test_compile.py`
  - `test_deepagents_setup.md`

### Fixed

- 🐛 Fixed structured output parameter conflicts with VLLM
- 🐛 Improved handling of `enable_thinking` with structured output
- 🐛 Fixed agent compatibility issues with LangChain 1.0
- 🐛 Resolved parameter serialization issues in agent execution

### Breaking Changes

⚠️ **Migration Required**: This release contains breaking changes for users of version 0.0.x

#### Agent API Changes
```python
# Old (0.3.x) - DeepAgent
from deepagents import DeepAgent
agent = DeepAgent(llm=llm, tools=[...])

# New (1.0.x) - LangChain create_agent
from langchain.agents import create_agent
agent = create_agent(model=llm, tools=[...])
```

#### Structured Output with Agents
```python
# Old (0.3.x)
structured_llm = llm.with_structured_output(schema=MySchema)

# New (1.0.x) - With agents
from langchain.agents.structured_output import ProviderStrategy
agent = create_agent(
    model=llm,
    response_format=ProviderStrategy(MySchema)
)
```

### Known Limitations

1. **Structured Output**: Must explicitly use `ProviderStrategy` as ChatQwenVllm is not automatically recognized by LangChain's auto-detection
2. **VLLM Constraints**: Cannot use `guided_json` with `enable_thinking` or `tools` simultaneously
3. **Test Coverage**: One integration test (`test_structured_output_with_lists`) may occasionally fail due to model JSON generation issues

### Migration Guide

See [LANGCHAIN_V1_MIGRATION.md](LANGCHAIN_V1_MIGRATION.md) for detailed migration instructions.

---

## [0.0.7] - 2024-XX-XX

### Previous releases based on LangChain 0.3.x

For historical changes, please refer to git history.

---

[1.0.0]: https://github.com/yourusername/langchain-qwq-vllm/compare/v0.0.7...v1.0.0

