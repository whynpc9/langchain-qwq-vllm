"""Test ChatQwQ chat model."""

from typing import Type

from dotenv import load_dotenv
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_qwq.chat_models import ChatQwen, ChatQwQ

load_dotenv()


class TestChatQwQIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatQwQ]:
        return ChatQwQ

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "qwq-plus",
        }

    @property
    def has_tool_choice(self) -> bool:
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        return False


class TestChatQwenIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatQwen]:
        return ChatQwen

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "qwen-plus-latest",
        }

    @property
    def has_tool_choice(self) -> bool:
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        return False
