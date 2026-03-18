"""Validate LangChain 1.x create_agent() compatibility across vLLM endpoints.

This script requires Python 3.10+ and a LangChain 1.x environment.
Example:

    PYTHONPATH=/tmp/langchain1py311:. python3.11 \
        tests/scripts/validate_create_agent_endpoints.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

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


class CalcResult(BaseModel):
    problem: str = Field(description="Original problem")
    final_answer: float = Field(description="Final answer")


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


@tool
def calculator(operation: str, a: float, b: float) -> str:
    """Perform basic math operations."""
    if operation == "multiply":
        return str(a * b)
    if operation == "add":
        return str(a + b)
    return "unsupported"


def _run_case(fn: Any) -> dict[str, Any]:
    try:
        return {"ok": True, "result": fn()}
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }


def _make_llm(
    config: EndpointConfig,
    *,
    enable_thinking: bool,
    max_tokens: int,
) -> ChatQwenVllm:
    return ChatQwenVllm(
        model=config.model,
        base_url=config.base_url,
        temperature=0,
        max_tokens=max_tokens,
        timeout=45,
        max_retries=0,
        enable_thinking=enable_thinking,
        vllm_mode=config.vllm_mode,
    )


def validate_endpoint(config: EndpointConfig) -> dict[str, Any]:
    def tools_only() -> dict[str, Any]:
        agent = create_agent(
            model=_make_llm(config, enable_thinking=True, max_tokens=32768),
            tools=[calculator],
            system_prompt="Use calculator for all calculations.",
        )
        out = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Use calculator to multiply 6 and 7.")
                ]
            }
        )
        return {
            "message_count": len(out["messages"]),
            "final_content": getattr(out["messages"][-1], "content", None),
            "tool_call_messages": [
                msg.tool_calls
                for msg in out["messages"]
                if getattr(msg, "tool_calls", None)
            ],
        }

    def provider_only() -> dict[str, Any]:
        agent = create_agent(
            model=_make_llm(config, enable_thinking=True, max_tokens=512),
            tools=[],
            system_prompt="Extract contact info.",
            response_format=ProviderStrategy(ContactInfo),
        )
        out = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Extract contact info: John Doe, john@example.com"
                    )
                ]
            }
        )
        response = out.get("structured_response")
        return response.model_dump() if hasattr(response, "model_dump") else response

    def toolstrategy_with_tools() -> dict[str, Any]:
        agent = create_agent(
            model=_make_llm(config, enable_thinking=True, max_tokens=32768),
            tools=[calculator],
            system_prompt="Use calculator, then return structured output.",
            response_format=ToolStrategy(
                CalcResult,
                tool_message_content="done",
            ),
        )
        out = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Use calculator to multiply 6 and 7, then return "
                            "the answer in structured form."
                        )
                    )
                ]
            }
        )
        response = out.get("structured_response")
        return response.model_dump() if hasattr(response, "model_dump") else response

    def tools_only_thinking_off() -> str | None:
        agent = create_agent(
            model=_make_llm(config, enable_thinking=False, max_tokens=32768),
            tools=[calculator],
            system_prompt="Use calculator for all calculations.",
        )
        out = agent.invoke(
            {
                "messages": [
                    HumanMessage(content="Use calculator to multiply 6 and 7.")
                ]
            }
        )
        return getattr(out["messages"][-1], "content", None)

    def toolstrategy_with_tools_thinking_off() -> dict[str, Any]:
        agent = create_agent(
            model=_make_llm(config, enable_thinking=False, max_tokens=32768),
            tools=[calculator],
            system_prompt="Use calculator, then return structured output.",
            response_format=ToolStrategy(
                CalcResult,
                tool_message_content="done",
            ),
        )
        out = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Use calculator to multiply 6 and 7, then return "
                            "the answer in structured form."
                        )
                    )
                ]
            }
        )
        response = out.get("structured_response")
        return response.model_dump() if hasattr(response, "model_dump") else response

    return {
        "config": asdict(config),
        "tools_only": _run_case(tools_only),
        "provider_only": _run_case(provider_only),
        "toolstrategy_with_tools": _run_case(toolstrategy_with_tools),
        "tools_only_thinking_off": _run_case(tools_only_thinking_off),
        "toolstrategy_with_tools_thinking_off": _run_case(
            toolstrategy_with_tools_thinking_off
        ),
    }


def main() -> None:
    results = {
        config.name: validate_endpoint(config)
        for config in ENDPOINTS
    }
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
