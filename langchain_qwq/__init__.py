from importlib import metadata

from langchain_qwq.chat_models import ChatQwen, ChatQwQ
from langchain_qwq.chat_models_vllm import ChatQwenVllm

try:
    __version__ = metadata.version(__package__)  # type:ignore
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatQwQ",
    "ChatQwen",
    "ChatQwenVllm",
    "__version__",
]
