"""Integration for QwQ and Qwen series chat models with VLLM backend"""

from copy import deepcopy
import json_repair as json
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils import from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field

from .chat_models import ChatQwen


DEFAULT_VLLM_API_BASE = "http://localhost:8000/v1"

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM], Type]
_DictOrPydantic = Union[Dict, _BM]


class ChatQwenVllm(ChatQwen):
    """Qwen series models integration with VLLM backend.

    This class extends ChatQwen to support VLLM (vLLM) as the backend inference engine
    for improved performance and efficiency when running Qwen models locally or in
    production environments.

    Setup:
        Install ``langchain-qwq`` and ``vllm``, then configure your VLLM 
        server endpoint.

        .. code-block:: bash

            pip install -U langchain-qwq vllm
            # Start VLLM server with a Qwen model
            vllm serve qwen/Qwen3-8B-Instruct --port 8000

    Key init args — completion params:
        model: str
            Name of Qwen model to use with VLLM, e.g.
            "qwen/Qwen3-8B-Instruct".
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    Key init args — client params:
        timeout: Optional[float]
            Timeout for requests.
        max_retries: int
            Max number of retries.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_qwq import ChatQwenVllm

            llm = ChatQwenVllm(
                model="qwen/Qwen3-8B-Instruct",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user " +
                 "sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.content, end="")

    Async:
        .. code-block:: python

            # Basic async invocation
            result = await llm.ainvoke(messages)

            # Stream response chunks
            async for chunk in llm.astream(messages):
                print(chunk.content, end="")

            # Batch processing of multiple message sets
            results = await llm.abatch([messages1, messages2])

    """

    api_base: str = Field(
        default_factory=from_env("VLLM_API_BASE", default=DEFAULT_VLLM_API_BASE),
        alias="base_url",
    )

    model_name: str = Field(default="Qwen/Qwen3-32B", alias="model")
    """The name of the Qwen model to use with VLLM"""

    enable_thinking: Optional[bool] = Field(default=True)
    """Whether to ask the vLLM chat template to keep reasoning enabled."""

    vllm_mode: Literal["legacy", "modern", "auto"] = Field(default="legacy")
    """How ChatQwenVllm should shape payloads for the target vLLM deployment."""

    modern_max_tokens_limit: int = Field(default=4096)
    """Fallback cap for modern vLLM tool/agent/structured-output requests."""

    def __getattribute__(self, name: str) -> Any:
        """Override to handle with_structured_output calls from RunnableBinding."""
        # Get the actual attribute first
        attr = super().__getattribute__(name)
        
        # Special handling for with_structured_output when called from RunnableBinding
        if name == "with_structured_output":
            import inspect
            
            # Check if we're being called from a RunnableBinding's __getattr__
            frame = inspect.currentframe()
            try:
                # Look for a RunnableBinding in the call stack
                calling_frame = frame.f_back
                if calling_frame and calling_frame.f_code.co_name == "__getattr__":
                    frame_locals = calling_frame.f_locals
                    if ('self' in frame_locals and 
                        hasattr(frame_locals['self'], 'kwargs') and
                        hasattr(frame_locals['self'], 'bound')):
                        binding = frame_locals['self']
                        if binding.bound is self and binding.kwargs:
                            # Store the binding kwargs temporarily
                            self._temp_binding_kwargs = binding.kwargs.copy()
            finally:
                del frame
        
        return attr

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-qwen"
    
    def _supports_structured_output(self) -> bool:
        """Indicate that this model supports structured output on vLLM backends."""
        return True

    def _check_need_stream(self) -> bool:
        """VLLM can handle both streaming and non-streaming efficiently."""
        return False

    def _support_tool_choice(self) -> bool:
        """VLLM backend supports tool choice functionality."""
        return True

    def _resolved_vllm_mode(self) -> Literal["legacy", "modern"]:
        """Resolve the vLLM compatibility mode.

        ``auto`` intentionally falls back to ``legacy`` to preserve the historical
        behavior of this integration unless the caller opts into ``modern``.
        """
        if self.vllm_mode == "auto":
            return "legacy"
        return self.vllm_mode

    def _build_extra_body(
        self, extra_body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build a request-local ``extra_body`` without leaking state across calls."""
        request_extra_body = deepcopy(extra_body or self.extra_body or {})
        request_extra_body.pop("enable_thinking", None)
        if self.enable_thinking is not None:
            chat_template_kwargs = request_extra_body.setdefault(
                "chat_template_kwargs", {}
            )
            chat_template_kwargs["enable_thinking"] = self.enable_thinking
        if self.thinking_budget is not None:
            request_extra_body["thinking_budget"] = self.thinking_budget
        return request_extra_body

    def _consume_temp_binding_kwargs(self) -> Dict[str, Any]:
        """Consume kwargs captured from a surrounding RunnableBinding."""
        bind_kwargs: Dict[str, Any] = {}
        if hasattr(self, "_temp_binding_kwargs"):
            bind_kwargs.update(self._temp_binding_kwargs)
            delattr(self, "_temp_binding_kwargs")
        return bind_kwargs

    @staticmethod
    def _set_enable_thinking(
        extra_body: Dict[str, Any], enabled: Optional[bool]
    ) -> Dict[str, Any]:
        """Set ``enable_thinking`` on request-local chat template kwargs."""
        if enabled is None:
            return extra_body

        chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
        chat_template_kwargs["enable_thinking"] = enabled
        return extra_body

    @staticmethod
    def _extract_guided_json_schema(response_format: Any) -> Optional[Dict[str, Any]]:
        """Extract a plain JSON schema from an OpenAI-style response_format value."""
        from langchain_core.utils.function_calling import convert_to_json_schema
        from langchain_openai.chat_models.base import _is_pydantic_class

        if _is_pydantic_class(response_format):
            return cast(Dict[str, Any], convert_to_json_schema(response_format))

        if not isinstance(response_format, dict):
            return None

        if response_format.get("type") != "json_schema":
            return None

        json_schema_data = response_format.get("json_schema", {})
        schema = json_schema_data.get("schema")
        if schema:
            return cast(Dict[str, Any], schema)

        if isinstance(json_schema_data, dict) and any(
            key in json_schema_data for key in ("type", "properties", "required")
        ):
            return cast(Dict[str, Any], json_schema_data)

        return None

    @staticmethod
    def _iter_structured_output_candidates(message: AIMessage) -> list[str]:
        """Collect candidate JSON strings from message content and reasoning."""
        candidates: list[str] = []

        content = message.content
        if isinstance(content, str) and content.strip():
            candidates.append(content)

        for key in ("reasoning_content", "reasoning"):
            value = message.additional_kwargs.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value)

        return candidates

    def _fallback_parse_structured_output(
        self,
        message: AIMessage,
        schema: _DictOrPydanticClass,
        *,
        is_pydantic_schema: bool,
    ) -> Any:
        """Parse structured output from content or reasoning when native parsed is absent."""
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_openai.chat_models.base import OpenAIRefusalError

        if parsed := message.additional_kwargs.get("parsed"):
            if is_pydantic_schema and isinstance(parsed, dict):
                return cast(Type[_BM], schema).model_validate(parsed)
            return parsed

        if refusal := message.additional_kwargs.get("refusal"):
            raise OpenAIRefusalError(refusal)

        if message.tool_calls:
            return None

        parser = JsonOutputParser()
        last_error: Optional[Exception] = None
        for candidate in self._iter_structured_output_candidates(message):
            try:
                if is_pydantic_schema:
                    data = json.loads(candidate)
                    return cast(Type[_BM], schema).model_validate(data)
                return parser.parse(candidate)
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        if last_error is not None:
            raise ValueError(
                "Structured output fallback could not parse JSON from either "
                "message content or reasoning_content."
            ) from last_error

        raise ValueError(
            "Structured output response did not contain parsable content, "
            "reasoning_content, parsed output, refusal, or tool calls."
        )

    def _maybe_promote_reasoning_json(self, message: AIMessage) -> None:
        """Promote JSON emitted in reasoning_content into message content.

        Some vLLM+Qwen combinations emit provider-structured JSON in
        ``reasoning_content`` while leaving ``content`` empty. LangChain 1.x
        ProviderStrategy parses message content directly, so normalize that shape
        here when the reasoning payload is valid JSON.
        """
        if message.tool_calls:
            return

        content = message.content if isinstance(message.content, str) else None
        if content and content.strip():
            return

        if message.additional_kwargs.get("parsed") is not None:
            return

        for candidate in self._iter_structured_output_candidates(message):
            stripped = candidate.strip()
            if not stripped.startswith(("{", "[")):
                continue

            try:
                parsed = json.loads(candidate)
            except Exception:  # noqa: BLE001
                continue

            message.content = candidate
            message.additional_kwargs["parsed"] = parsed
            return

    def _is_structured_output_payload(self, payload: Dict[str, Any]) -> bool:
        """Check whether the outgoing request is a structured-output request."""
        extra_body = payload.get("extra_body")
        return "response_format" in payload or (
            isinstance(extra_body, dict) and "guided_json" in extra_body
        )

    def _apply_modern_token_cap(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp oversized completion budgets for modern vLLM agent/tool requests."""
        if self._resolved_vllm_mode() != "modern":
            return payload

        if not (
            "tools" in payload
            or self._is_structured_output_payload(payload)
        ):
            return payload

        limit = self.modern_max_tokens_limit
        if limit <= 0:
            return payload

        token_keys = ("max_completion_tokens", "max_tokens")
        has_budget = False
        for key in token_keys:
            value = payload.get(key)
            if isinstance(value, int):
                payload[key] = min(value, limit)
                has_budget = True
            elif value is not None:
                has_budget = True

        if not has_budget:
            payload["max_tokens"] = limit

        return payload

    def _apply_structured_output_compatibility(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt structured-output payloads for legacy vs. modern vLLM."""
        if "response_format" not in payload:
            return payload

        response_format = payload["response_format"]
        mode = self._resolved_vllm_mode()

        if mode == "modern":
            extra_body = deepcopy(payload.get("extra_body") or {})
            payload["extra_body"] = self._set_enable_thinking(extra_body, False)
            return payload

        schema = self._extract_guided_json_schema(response_format)
        if schema is None:
            return payload

        payload["extra_body"] = {"guided_json": schema}
        del payload["response_format"]

        # Legacy vLLM cannot reliably combine guided_json with tools or tool-choice.
        for key in ("tools", "parallel_tool_calls", "tool_choice"):
            payload.pop(key, None)

        return payload
    
    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """Override to adapt payloads for legacy and modern vLLM deployments."""
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        payload = self._apply_structured_output_compatibility(payload)
        payload = self._apply_modern_token_cap(payload)

        return payload

    def _create_chat_result(
        self,
        response: Any,
        generation_info: Optional[Dict] = None,
    ) -> ChatResult:
        """Normalize structured-output responses before LangChain parses them."""
        result = super()._create_chat_result(response, generation_info)
        if result.generations:
            message = result.generations[0].message
            if isinstance(message, AIMessage):
                self._maybe_promote_reasoning_json(message)
        return result

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling VLLM API."""
        self.extra_body = self._build_extra_body()
        params = BaseChatOpenAI._default_params.fget(self)

        return params

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal["json_schema"] = "json_schema",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Enable structured output for VLLM backend using guided_json.
        
        This method supports only json_schema method and uses vLLM's guided_json
        parameter instead of modifying system prompts.
        
        Args:
            schema: JSON schema or Pydantic model class for structured output
            method: Only "json_schema" is supported for vLLM backend
            include_raw: Whether to include raw response in output
            **kwargs: Additional arguments (currently unused)
            
        Returns:
            Runnable that outputs structured data according to the schema
            
        Raises:
            ValueError: If method is not "json_schema" or if schema is not provided
        """
        from langchain_core.output_parsers import (
            JsonOutputParser,
            PydanticOutputParser,
        )
        from langchain_core.runnables import RunnableLambda
        from langchain_core.utils.function_calling import convert_to_json_schema
        from langchain_openai.chat_models.base import _is_pydantic_class

        # Only support json_schema method for vLLM
        if method != "json_schema":
            raise ValueError(
                f"Method '{method}' is not supported. Only 'json_schema' is "
                f"supported for vLLM backend."
            )

        if schema is None:
            raise ValueError(
                "Schema must be provided when using vLLM structured output"
            )

        if self._resolved_vllm_mode() == "modern":
            strict = kwargs.pop("strict", None)
            is_pydantic_schema = _is_pydantic_class(schema)
            bound_llm: Any = self
            bind_kwargs = self._consume_temp_binding_kwargs()
            if bind_kwargs:
                bound_llm = self.bind(**bind_kwargs)
            raw_chain = BaseChatOpenAI.with_structured_output(
                bound_llm,
                schema=schema,
                method="json_schema",
                include_raw=True,
                strict=strict,
                **kwargs,
            )

            def process_with_fallback(x: Any) -> Any:
                result = cast(Dict[str, Any], raw_chain.invoke(x))
                if result.get("parsing_error") is not None and result.get("raw") is not None:
                    try:
                        result["parsed"] = self._fallback_parse_structured_output(
                            cast(AIMessage, result["raw"]),
                            schema,
                            is_pydantic_schema=is_pydantic_schema,
                        )
                        result["parsing_error"] = None
                    except Exception as exc:  # noqa: BLE001
                        result["parsing_error"] = exc

                if include_raw:
                    return result
                if result.get("parsing_error") is not None:
                    raise cast(Exception, result["parsing_error"])
                return result.get("parsed")

            async def aprocess_with_fallback(x: Any) -> Any:
                result = cast(Dict[str, Any], await raw_chain.ainvoke(x))
                if result.get("parsing_error") is not None and result.get("raw") is not None:
                    try:
                        result["parsed"] = self._fallback_parse_structured_output(
                            cast(AIMessage, result["raw"]),
                            schema,
                            is_pydantic_schema=is_pydantic_schema,
                        )
                        result["parsing_error"] = None
                    except Exception as exc:  # noqa: BLE001
                        result["parsing_error"] = exc

                if include_raw:
                    return result
                if result.get("parsing_error") is not None:
                    raise cast(Exception, result["parsing_error"])
                return result.get("parsed")

            return RunnableLambda(
                process_with_fallback,
                afunc=aprocess_with_fallback,
            )

        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")

        is_pydantic_schema = _is_pydantic_class(schema)

        # Convert schema to JSON schema format
        if schema and isinstance(schema, dict):
            # Ensure the schema dictionary has a 'title' for convert_to_json_schema
            if "title" not in schema:
                schema_for_conversion = schema.copy()
                schema_for_conversion["title"] = "extracted_data"
            else:
                schema_for_conversion = schema
            schema_dict = convert_to_json_schema(schema_for_conversion)
        elif is_pydantic_schema:
            schema_dict = convert_to_json_schema(schema)
        else:
            schema_dict = schema

        # Use guided_json parameter instead of modifying system prompt
        extra_body = self._build_extra_body()
        extra_body.pop("chat_template_kwargs", None)
        extra_body["guided_json"] = schema_dict
        
        # Start with existing kwargs (like tools) from RunnableBinding
        bind_kwargs = self._consume_temp_binding_kwargs()
        
        # Merge existing extra_body with guided_json
        if 'extra_body' in bind_kwargs:
            existing_extra_body = bind_kwargs.get('extra_body', {}) or {}
            # Merge and ensure guided_json takes precedence
            extra_body = {**existing_extra_body, **extra_body}
            extra_body.pop("chat_template_kwargs", None)
        
        # Set final bind arguments
        bind_kwargs.update({
            "extra_body": extra_body,
            "ls_structured_output_format": {
                "kwargs": {"method": method},
                "schema": schema,
            },
        })
        
        llm = self.bind(**bind_kwargs)

        output_parser = (
            PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
            if is_pydantic_schema
            else JsonOutputParser()
        )

        def parse_output(input_: AIMessage) -> Any:
            if isinstance(input_.content, str):
                return output_parser.parse(input_.content)
            else:
                return input_.content

        if include_raw:
            # Include raw output in the result
            def process_with_raw(x: Any) -> Dict[str, Any]:
                raw_output = cast(AIMessage, llm.invoke(x))
                try:
                    parsed = parse_output(raw_output)
                    return {
                        "raw": raw_output,
                        "parsed": parsed,
                        "parsing_error": None,
                    }
                except Exception as e:
                    return {
                        "raw": raw_output,
                        "parsed": None,
                        "parsing_error": e,
                    }

            async def aprocess_with_raw(x: Any) -> Dict[str, Any]:
                raw_output = cast(AIMessage, await llm.ainvoke(x))
                try:
                    parsed = parse_output(raw_output)
                    return {
                        "raw": raw_output,
                        "parsed": parsed,
                        "parsing_error": None,
                    }
                except Exception as e:
                    return {
                        "raw": raw_output,
                        "parsed": None,
                        "parsing_error": e,
                    }

            chain = RunnableLambda(process_with_raw, afunc=aprocess_with_raw)  # type: ignore
        else:
            # Only return parsed output
            def process_without_raw(x: Any) -> Any:
                raw_output = cast(AIMessage, llm.invoke(x))
                output = parse_output(raw_output)
                return output

            async def aprocess_without_raw(x: Any) -> Any:
                raw_output = cast(AIMessage, await llm.ainvoke(x))
                output = parse_output(raw_output)
                return output

            chain = RunnableLambda(process_without_raw, afunc=aprocess_without_raw)  # type: ignore

        return chain
