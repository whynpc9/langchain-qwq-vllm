"""Integration for QwQ and Qwen series chat models with VLLM backend"""

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
from langchain_core.runnables import Runnable
from langchain_core.utils import from_env
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

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-qwen"

    def _check_need_stream(self) -> bool:
        """VLLM can handle both streaming and non-streaming efficiently."""
        return False

    def _support_tool_choice(self) -> bool:
        """VLLM backend supports tool choice functionality."""
        return True

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling VLLM API."""
        # Override the base implementation to include vllm-specific
        # thinking-related params
        if self.extra_body is None:
            self.extra_body = {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            }
        else:
            self.extra_body["chat_template_kwargs"]["enable_thinking"] = (
                self.enable_thinking
            )

        params = super()._default_params

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

        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")

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
        extra_body = self.extra_body or {}
        extra_body["guided_json"] = schema_dict
        
        # Start with existing kwargs (like tools) from RunnableBinding
        bind_kwargs = {}
        
        # CRITICAL: For vLLM integration, we need to ensure tools are preserved
        # The challenge is that when called on a RunnableBinding, this method
        # executes on the bound ChatQwenVllm object, losing access to the binding kwargs.
        # Solution: Pass binding kwargs via kwargs parameter from RunnableBinding.
        
        if hasattr(self, 'kwargs') and self.kwargs:
            # Direct call on RunnableBinding
            bind_kwargs.update(self.kwargs)
        elif '_bound_kwargs' in kwargs:
            # Binding kwargs passed explicitly (our solution)
            bind_kwargs.update(kwargs.pop('_bound_kwargs'))
        
        # If no binding kwargs found, this might be a delegated call
        # For now, we proceed with empty bind_kwargs, but the real fix
        # would be at the RunnableBinding level
        
        # Merge existing extra_body with guided_json
        if 'extra_body' in bind_kwargs:
            existing_extra_body = bind_kwargs.get('extra_body', {}) or {}
            # Merge and ensure guided_json takes precedence
            extra_body = {**existing_extra_body, **extra_body}
        
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
