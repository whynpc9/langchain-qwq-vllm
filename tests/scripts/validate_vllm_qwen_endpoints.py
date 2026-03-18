"""Validate ChatQwenVllm behavior against documented Qwen/vLLM endpoints."""

from __future__ import annotations

import json
import os
import sys
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_qwq.chat_models_vllm import ChatQwenVllm


os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-vllm")


@dataclass
class EndpointConfig:
    name: str
    base_url: str
    model: str
    vllm_mode: str


class ContactInfo(BaseModel):
    name: str = Field(description="Person name")
    email: str = Field(description="Email address")


ENDPOINTS = [
    EndpointConfig(
        name="old_vllm_qwen3_32b",
        base_url="http://172.16.4.232:8000/v1",
        model="Qwen/Qwen3-32B",
        vllm_mode="legacy",
    ),
    EndpointConfig(
        name="new_vllm_qwen3_32b",
        base_url="http://172.16.4.232:8301/v1",
        model="Qwen/Qwen3-32B",
        vllm_mode="modern",
    ),
    EndpointConfig(
        name="new_vllm_qwen3_5_27b",
        base_url="http://172.16.4.232:8302/v1",
        model="Qwen/Qwen3.5-27B",
        vllm_mode="modern",
    ),
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _exc_dict(exc: Exception) -> Dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(limit=5),
    }


def _fetch_models(base_url: str) -> Dict[str, Any]:
    try:
        request = urllib.request.Request(f"{base_url}/models")
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
        return {"ok": True, "status": 200, "body": json.loads(body)}
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": exc.code,
            "body": exc.read().decode("utf-8", errors="replace"),
        }
    except Exception as exc:
        return {"ok": False, "error": _exc_dict(exc)}


def _message_snapshot(message: Any) -> Dict[str, Any]:
    content = getattr(message, "content", None)
    if isinstance(content, list):
        content = _json_safe(content)

    return {
        "content": content,
        "tool_calls": _json_safe(getattr(message, "tool_calls", None)),
        "additional_kwargs": _json_safe(getattr(message, "additional_kwargs", {})),
        "response_metadata": _json_safe(getattr(message, "response_metadata", {})),
    }


def _thinking_probe(config: EndpointConfig, enabled: bool) -> Dict[str, Any]:
    llm = ChatQwenVllm(
        model=config.model,
        base_url=config.base_url,
        temperature=0,
        max_tokens=128,
        max_retries=0,
        timeout=20,
        enable_thinking=enabled,
        vllm_mode=config.vllm_mode,
    )
    payload = llm._get_request_payload("Reply with exactly ok.")

    try:
        message = llm.invoke("Reply with exactly ok.")
        return {
            "ok": True,
            "payload": {
                "max_tokens": payload.get("max_tokens"),
                "extra_body": payload.get("extra_body"),
            },
            "message": _message_snapshot(message),
        }
    except Exception as exc:
        return {
            "ok": False,
            "payload": {
                "max_tokens": payload.get("max_tokens"),
                "extra_body": payload.get("extra_body"),
            },
            "error": _exc_dict(exc),
        }


def _structured_output_probe(config: EndpointConfig) -> Dict[str, Any]:
    llm = ChatQwenVllm(
        model=config.model,
        base_url=config.base_url,
        temperature=0,
        max_tokens=512,
        max_retries=0,
        timeout=30,
        enable_thinking=True,
        vllm_mode=config.vllm_mode,
    )

    payload = llm._get_request_payload(
        "Extract contact info: John Doe, john@example.com",
        response_format=ContactInfo,
    )
    structured_llm = llm.with_structured_output(ContactInfo, include_raw=True)

    try:
        result = structured_llm.invoke(
            "Extract contact info: John Doe, john@example.com"
        )
        raw = result.get("raw")
        parsed = result.get("parsed")
        parsing_error = result.get("parsing_error")
        return {
            "ok": parsing_error is None and parsed is not None,
            "payload": {
                "max_tokens": payload.get("max_tokens"),
                "response_format": _json_safe(payload.get("response_format")),
                "extra_body": _json_safe(payload.get("extra_body")),
            },
            "raw": _message_snapshot(raw) if raw is not None else None,
            "parsed": _json_safe(parsed),
            "parsing_error": str(parsing_error) if parsing_error else None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "payload": {
                "max_tokens": payload.get("max_tokens"),
                "response_format": _json_safe(payload.get("response_format")),
                "extra_body": _json_safe(payload.get("extra_body")),
            },
            "error": _exc_dict(exc),
        }


def _tool_call_probe(config: EndpointConfig, enabled: bool) -> Dict[str, Any]:
    def lookup_contact(name: str) -> str:
        """Look up a contact by name."""
        return f"{name}: john@example.com"

    llm = ChatQwenVllm(
        model=config.model,
        base_url=config.base_url,
        temperature=0,
        max_tokens=32768,
        max_retries=0,
        timeout=30,
        enable_thinking=enabled,
        vllm_mode=config.vllm_mode,
    )
    runnable = llm.bind_tools(
        [lookup_contact],
        tool_choice="lookup_contact",
        parallel_tool_calls=False,
    )
    payload = llm._get_request_payload(
        "Call lookup_contact for John Doe.",
        tools=runnable.kwargs["tools"],
        tool_choice=runnable.kwargs["tool_choice"],
        parallel_tool_calls=runnable.kwargs["parallel_tool_calls"],
    )

    try:
        message = runnable.invoke("Call lookup_contact for John Doe.")
        return {
            "ok": bool(getattr(message, "tool_calls", None)),
            "payload": {
                "max_tokens": payload.get("max_tokens"),
                "tool_choice": payload.get("tool_choice"),
                "parallel_tool_calls": payload.get("parallel_tool_calls"),
                "extra_body": _json_safe(payload.get("extra_body")),
            },
            "message": _message_snapshot(message),
        }
    except Exception as exc:
        return {
            "ok": False,
            "payload": {
                "max_tokens": payload.get("max_tokens"),
                "tool_choice": payload.get("tool_choice"),
                "parallel_tool_calls": payload.get("parallel_tool_calls"),
                "extra_body": _json_safe(payload.get("extra_body")),
            },
            "error": _exc_dict(exc),
        }


def validate_endpoint(config: EndpointConfig) -> Dict[str, Any]:
    return {
        "config": asdict(config),
        "models": _fetch_models(config.base_url),
        "thinking_on": _thinking_probe(config, enabled=True),
        "thinking_off": _thinking_probe(config, enabled=False),
        "structured_output": _structured_output_probe(config),
        "tool_calling_thinking_on": _tool_call_probe(config, enabled=True),
        "tool_calling_thinking_off": _tool_call_probe(config, enabled=False),
    }


def main() -> None:
    results = {
        config.name: validate_endpoint(config)
        for config in ENDPOINTS
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
