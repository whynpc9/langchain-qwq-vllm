"""Unit tests for ChatQwenVllm payload shaping."""

import os

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from langchain_qwq.chat_models_vllm import ChatQwenVllm


os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-vllm")


def _structured_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "contact_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": ["name", "email"],
            },
        },
    }


def _tool_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup_contact",
                "description": "Look up a contact by name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
        }
    ]


def test_legacy_payload_uses_guided_json_and_drops_tools() -> None:
    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        base_url="http://localhost:8000/v1",
        max_tokens=32768,
        enable_thinking=True,
        vllm_mode="legacy",
    )

    payload = llm._get_request_payload(
        "Extract John Doe <john@example.com>.",
        response_format=_structured_response_format(),
        tools=_tool_schema(),
        parallel_tool_calls=True,
    )

    assert "response_format" not in payload
    assert payload["extra_body"] == {
        "guided_json": _structured_response_format()["json_schema"]["schema"]
    }
    assert "tools" not in payload
    assert "parallel_tool_calls" not in payload
    assert payload["max_tokens"] == 32768


def test_legacy_payload_converts_pydantic_response_format() -> None:
    class ContactInfo(BaseModel):
        name: str
        email: str

    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        base_url="http://localhost:8000/v1",
        enable_thinking=True,
        vllm_mode="legacy",
    )

    payload = llm._get_request_payload(
        "Extract contact info.",
        response_format=ContactInfo,
    )

    assert "response_format" not in payload
    assert payload["extra_body"]["guided_json"]["title"] == "ContactInfo"
    assert "name" in payload["extra_body"]["guided_json"]["properties"]
    assert "email" in payload["extra_body"]["guided_json"]["properties"]


def test_modern_payload_keeps_native_response_format_and_caps_tokens() -> None:
    llm = ChatQwenVllm(
        model="Qwen/Qwen3.5-27B",
        base_url="http://localhost:8302/v1",
        max_tokens=32768,
        enable_thinking=True,
        vllm_mode="modern",
    )

    response_format = _structured_response_format()
    payload = llm._get_request_payload(
        "Extract John Doe <john@example.com>.",
        response_format=response_format,
        tools=_tool_schema(),
        parallel_tool_calls=True,
    )

    assert payload["response_format"] == response_format
    assert payload["tools"] == _tool_schema()
    assert payload["parallel_tool_calls"] is True
    assert payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert payload["max_tokens"] == 4096


def test_modern_token_cap_only_applies_to_tool_or_structured_requests() -> None:
    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        base_url="http://localhost:8301/v1",
        max_tokens=32768,
        enable_thinking=True,
        vllm_mode="modern",
    )

    plain_payload = llm._get_request_payload("Reply with ok.")
    tool_payload = llm._get_request_payload(
        "Use the tool.",
        tools=_tool_schema(),
        parallel_tool_calls=True,
    )

    assert plain_payload["max_tokens"] == 32768
    assert tool_payload["max_tokens"] == 4096


def test_modern_structured_output_fallback_parses_reasoning_content() -> None:
    class ContactInfo(BaseModel):
        name: str
        email: str

    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        base_url="http://localhost:8301/v1",
        vllm_mode="modern",
    )

    message = AIMessage(
        content="",
        additional_kwargs={
            "parsed": None,
            "refusal": None,
            "reasoning_content": '{"name": "John Doe", "email": "john@example.com"}',
        },
    )

    parsed = llm._fallback_parse_structured_output(
        message,
        ContactInfo,
        is_pydantic_schema=True,
    )

    assert isinstance(parsed, ContactInfo)
    assert parsed.name == "John Doe"
    assert parsed.email == "john@example.com"


def test_modern_structured_output_fallback_parses_content_json() -> None:
    llm = ChatQwenVllm(
        model="Qwen/Qwen3.5-27B",
        base_url="http://localhost:8302/v1",
        vllm_mode="modern",
    )

    message = AIMessage(
        content='{"name": "John Doe", "email": "john@example.com"}',
        additional_kwargs={"parsed": None, "refusal": None},
    )

    parsed = llm._fallback_parse_structured_output(
        message,
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["name", "email"],
        },
        is_pydantic_schema=False,
    )

    assert parsed == {"name": "John Doe", "email": "john@example.com"}


def test_reasoning_json_is_promoted_into_content_for_provider_strategy() -> None:
    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        base_url="http://localhost:8301/v1",
        vllm_mode="modern",
    )

    message = AIMessage(
        content="",
        additional_kwargs={
            "reasoning_content": '{"name": "John Doe", "email": "john@example.com"}'
        },
    )

    llm._maybe_promote_reasoning_json(message)

    assert message.content == '{"name": "John Doe", "email": "john@example.com"}'
    assert message.additional_kwargs["parsed"] == {
        "name": "John Doe",
        "email": "john@example.com",
    }
