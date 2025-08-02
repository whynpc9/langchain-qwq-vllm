from typing import AsyncIterator, Iterator, Tuple

from langchain_core.messages import AIMessageChunk, BaseMessageChunk


def convert_reasoning_to_content(
    model_response: Iterator[BaseMessageChunk],
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> Iterator[BaseMessageChunk]:
    isfirst = True
    isend = True

    for chunk in model_response:
        if (
            isinstance(chunk, AIMessageChunk)
            and "reasoning_content" in chunk.additional_kwargs
        ):
            if isfirst:
                chunk.content = (
                    f"{think_tag[0]}{chunk.additional_kwargs['reasoning_content']}"
                )
                isfirst = False
            else:
                chunk.content = chunk.additional_kwargs["reasoning_content"]
        elif (
            isinstance(chunk, AIMessageChunk)
            and "reasoning_content" not in chunk.additional_kwargs
            and chunk.content
            and isend
        ):
            chunk.content = f"{think_tag[1]}{chunk.content}"
            isend = False
        yield chunk


async def aconvert_reasoning_to_content(
    amodel_response: AsyncIterator[BaseMessageChunk],
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> AsyncIterator[BaseMessageChunk]:
    isfirst = True
    isend = True
    async for chunk in amodel_response:
        if (
            isinstance(chunk, AIMessageChunk)
            and "reasoning_content" in chunk.additional_kwargs
        ):
            if isfirst:
                chunk.content = (
                    f"{think_tag[0]}{chunk.additional_kwargs['reasoning_content']}"
                )
                isfirst = False
            else:
                chunk.content = chunk.additional_kwargs["reasoning_content"]
        elif (
            isinstance(chunk, AIMessageChunk)
            and "reasoning_content" not in chunk.additional_kwargs
            and chunk.content
            and isend
        ):
            chunk.content = f"{think_tag[1]}{chunk.content}"
            isend = False
        yield chunk


enable_streaming_model = [
    "qwen3-235b-a22b",
    "qwen3-32b",
    "qwen3-30b-a3b",
    "qwen3-14b",
    "qwen3-8b",
    "qwen3-4b",
    "qwen3-1.7b",
    "qwen3-0.6b",
]


support_tool_choice_models = [
    "qwen3-235b-a22b-instruct-2507",
    "qwen3-30b-a3b-instruct-2507",
    "qwen3-coder-480b-a35b-instruct",
    "qwen3-coder-plus",
    "qwen3-coder-30b-a3b-instruct",
    "qwen-max",
    "qwen-max-latest",
    "qwen-plus",
    "qwen-plus-latest",
    "qwen-turbo",
    "qwen-turbo-latest",
    "qwen3-235b-a22b",
    "qwen3-32b",
    "qwen3-30b-a3b",
    "qwen3-14b",
    "qwen3-8b",
    "qwen3-4b",
    "qwen3-1.7b",
    "qwen3-0.6b",
    "qwen2.5-14b-instruct-1m",
    "qwen2.5-7b-instruct-1m",
    "qwen2.5-72b-instruct",
    "qwen2.5-32b-instruct",
    "qwen2.5-14b-instruct",
    "qwen2.5-7b-instruct",
    "qwen2.5-3b-instruct",
    "qwen2.5-1.5b-instruct",
    "qwen2.5-0.5b-instruct",
]


def _check_support_tool_choice(model: str) -> bool:
    if model in support_tool_choice_models:
        return True
    return False


model_not_support_json_mode = [
    "qwen3-235b-a22b-thinking-2507",
    "qwen3-30b-a3b-thinking-2507",
]
