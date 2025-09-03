"""Integration for QwQ and Qwen series chat models with VLLM backend"""

from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
)

from langchain_core.runnables import Runnable
from pydantic import Field

from .chat_models import ChatQwen

from langchain_core.utils import from_env

from langchain_openai.chat_models.base import BaseChatOpenAI


DEFAULT_VLLM_API_BASE = "http://localhost:8000/v1"


class ChatQwenVllm(ChatQwen):
    """Qwen series models integration with VLLM backend.

    This class extends ChatQwen to support VLLM (vLLM) as the backend inference engine
    for improved performance and efficiency when running Qwen models locally or in
    production environments.

    Setup:
        Install ``langchain-qwq`` and ``vllm``, then configure your VLLM server endpoint.

        .. code-block:: bash

            pip install -U langchain-qwq vllm
            # Start VLLM server with a Qwen model
            vllm serve qwen/Qwen3-8B-Instruct --port 8000

    Key init args — completion params:
        model: str
            Name of Qwen model to use with VLLM, e.g. "qwen/Qwen3-8B-Instruct".
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
                ("system", "You are a helpful translator. Translate the user sentence to French."),
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
        # Override the base implementation to not include thinking-related params
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
