"""Integration tests for ChatQwenVllm with VLLM backend."""

from typing import Type

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel

# Try to import langchain_tests, but provide fallback
try:
    from langchain_tests.integration_tests import ChatModelIntegrationTests
    LANGCHAIN_TESTS_AVAILABLE = True
except ImportError:
    LANGCHAIN_TESTS_AVAILABLE = False
    # Create a dummy base class when langchain_tests is not available
    class ChatModelIntegrationTests:
        pass

from langchain_qwq.chat_models_vllm import ChatQwenVllm

load_dotenv()


@pytest.mark.skipif(not LANGCHAIN_TESTS_AVAILABLE, reason="langchain_tests not available")
class TestChatQwenVllmIntegration(ChatModelIntegrationTests):
    """Standard integration tests for ChatQwenVllm."""
    
    @property
    def chat_model_class(self) -> Type[ChatQwenVllm]:
        return ChatQwenVllm

    @property
    def chat_model_params(self) -> dict:
        """Parameters for initializing ChatQwenVllm in tests."""
        return {
            "model": "Qwen/Qwen3-32B",
            "temperature": 0.1,
            "max_tokens": 100,
            "enable_thinking": True,
        }

    @property
    def has_tool_choice(self) -> bool:
        return True

    @property
    def supports_image_tool_message(self) -> bool:
        return False

    @property
    def supports_json_mode(self) -> bool:
        return True


class TestChatQwenVllmReasoningContent:
    """Test reasoning content handling for ChatQwenVllm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.llm = ChatQwenVllm(
            model="Qwen/Qwen3-32B",
            temperature=0.1,
            max_tokens=500,
            enable_thinking=True,
        )
    
    def test_basic_invoke_with_reasoning(self):
        """Test basic invocation and reasoning content access."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Solve this math problem: What is 15 + 27?"),
        ]
        
        response = self.llm.invoke(messages)
        
        # Check that we get a valid response
        assert response.content is not None
        
        # Check for reasoning content in additional_kwargs
        assert "reasoning_content" in response.additional_kwargs
        reasoning_content = response.additional_kwargs.get("reasoning_content", "")
        
        # Should have either content or reasoning content (or both)
        has_content = len(response.content) > 0
        has_reasoning = len(reasoning_content) > 0
        assert has_content or has_reasoning, f"Response should have content or reasoning. Content: '{response.content}', Reasoning: '{reasoning_content[:100]}...'"
    
    def test_stream_with_reasoning_content(self):
        """Test streaming with reasoning content handling."""
        messages = [
            SystemMessage(
                content="You are a helpful assistant that thinks step by step."
            ),
            HumanMessage(content="Explain how to calculate the area of a circle."),
        ]
        
        chunks = list(self.llm.stream(messages))
        
        # Should receive multiple chunks
        assert len(chunks) > 0
        
        # Check for reasoning content chunks
        reasoning_chunks = [
            chunk for chunk in chunks 
            if hasattr(chunk, 'additional_kwargs') 
            and "reasoning_content" in chunk.additional_kwargs
        ]
        
        # Check for content chunks
        content_chunks = [
            chunk for chunk in chunks 
            if hasattr(chunk, 'content') and chunk.content
        ]
        
        # Should have at least some content chunks or reasoning chunks
        assert len(content_chunks) > 0 or len(reasoning_chunks) > 0
        
        # Combine all content
        full_content = "".join(chunk.content for chunk in content_chunks)
        full_reasoning = "".join(
            chunk.additional_kwargs["reasoning_content"] 
            for chunk in reasoning_chunks
        )
        
        # Should have either content or reasoning
        assert len(full_content) > 0 or len(full_reasoning) > 0
    
    @pytest.mark.asyncio
    async def test_async_invoke_with_reasoning(self):
        """Test async invocation with reasoning content."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What are the main benefits of renewable energy?"),
        ]
        
        response = await self.llm.ainvoke(messages)
        
        # Check response content
        assert response.content is not None
        
        # Check for reasoning content
        assert "reasoning_content" in response.additional_kwargs
        reasoning_content = response.additional_kwargs.get("reasoning_content", "")
        
        # Should have either content or reasoning content (or both)
        has_content = len(response.content) > 0
        has_reasoning = len(reasoning_content) > 0
        assert has_content or has_reasoning, f"Response should have content or reasoning. Content: '{response.content}', Reasoning: '{reasoning_content[:100]}...'"
    
    @pytest.mark.asyncio
    async def test_async_stream_with_reasoning_content(self):
        """Test async streaming with reasoning content."""
        messages = [
            SystemMessage(
                content="You are a helpful assistant that explains concepts clearly."
            ),
            HumanMessage(
                content="Explain the concept of machine learning in simple terms."
            ),
        ]
        
        chunks = []
        async for chunk in self.llm.astream(messages):
            chunks.append(chunk)
        
        # Should receive multiple chunks
        assert len(chunks) > 0
        
        # Check for content chunks
        content_chunks = [
            chunk for chunk in chunks 
            if hasattr(chunk, 'content') and chunk.content
        ]
        
        # Check for reasoning chunks  
        reasoning_chunks = [
            chunk for chunk in chunks 
            if (
                hasattr(chunk, "additional_kwargs")
                and "reasoning_content" in chunk.additional_kwargs
            )
        ]
        
        # Should have at least some content chunks or reasoning chunks
        assert len(content_chunks) > 0 or len(reasoning_chunks) > 0
        
        # Combine all content
        full_content = "".join(chunk.content for chunk in content_chunks)
        full_reasoning = "".join(
            chunk.additional_kwargs["reasoning_content"] 
            for chunk in reasoning_chunks
        )
        
        # Should have either content or reasoning
        assert len(full_content) > 0 or len(full_reasoning) > 0
    
    def test_reasoning_content_structure_sync(self):
        """Test the structure of reasoning content in sync mode."""
        messages = [
            HumanMessage(
                content=(
                    "Analyze this problem: A train travels 120 km in 2 hours. "
                    "What is its speed?"
                )
            ),
        ]
        
        # Test with streaming to see the progressive reasoning
        chunks = list(self.llm.stream(messages))
        
        # Collect reasoning and content separately
        reasoning_parts = []
        content_parts = []
        
        for chunk in chunks:
            if (
                hasattr(chunk, "additional_kwargs")
                and "reasoning_content" in chunk.additional_kwargs
            ):
                reasoning_parts.append(
                    chunk.additional_kwargs["reasoning_content"]
                )
            elif hasattr(chunk, "content") and chunk.content:
                content_parts.append(chunk.content)
        
        # Content or reasoning should exist
        assert len(content_parts) > 0 or len(reasoning_parts) > 0, f"Should have either content or reasoning. Content parts: {len(content_parts)}, Reasoning parts: {len(reasoning_parts)}"
        
        # Combine content if available
        full_content = "".join(content_parts)
        full_reasoning = "".join(reasoning_parts)
        
        # Should have either content or reasoning (or both)
        assert len(full_content) > 0 or len(full_reasoning) > 0, f"Should have either content or reasoning. Content: '{full_content}', Reasoning: '{full_reasoning[:100]}...'"
        
        # Test with non-streaming to get final reasoning
        response = self.llm.invoke(messages)
        assert "reasoning_content" in response.additional_kwargs
    
    @pytest.mark.asyncio
    async def test_reasoning_content_structure_async(self):
        """Test the structure of reasoning content in async mode."""
        messages = [
            HumanMessage(
                content=(
                    "Compare the advantages and disadvantages of "
                    "solar vs wind energy."
                )
            ),
        ]
        
        # Test async streaming
        chunks = []
        async for chunk in self.llm.astream(messages):
            chunks.append(chunk)
        
        # Collect reasoning and content separately
        reasoning_parts = []
        content_parts = []
        
        for chunk in chunks:
            if (
                hasattr(chunk, "additional_kwargs")
                and "reasoning_content" in chunk.additional_kwargs
            ):
                reasoning_parts.append(
                    chunk.additional_kwargs["reasoning_content"]
                )
            elif hasattr(chunk, "content") and chunk.content:
                content_parts.append(chunk.content)
        
        # Content or reasoning should exist
        assert len(content_parts) > 0 or len(reasoning_parts) > 0, f"Should have either content or reasoning. Content parts: {len(content_parts)}, Reasoning parts: {len(reasoning_parts)}"
        
        # Combine content if available
        full_content = "".join(content_parts)
        full_reasoning = "".join(reasoning_parts)
        
        # Should have either content or reasoning (or both)
        assert len(full_content) > 0 or len(full_reasoning) > 0, f"Should have either content or reasoning. Content: '{full_content}', Reasoning: '{full_reasoning[:100]}...'"
        
        # Test async invoke
        response = await self.llm.ainvoke(messages)
        assert "reasoning_content" in response.additional_kwargs


class TestChatQwenVllmFeatures:
    """Test specific features of ChatQwenVllm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.llm = ChatQwenVllm(
            model="Qwen/Qwen3-32B",
            temperature=0.1,
            enable_thinking=True,
        )
    
    def test_tool_calling(self):
        """Test tool calling functionality."""
        @tool
        def get_weather(city: str) -> str:
            """Get the weather for a city."""
            return f"The weather in {city} is sunny with 22°C."
        
        llm_with_tools = self.llm.bind_tools([get_weather])
        
        response = llm_with_tools.invoke("What's the weather in Beijing?")
        
        # Should either have tool calls or mention the tool in content
        if hasattr(response, 'tool_calls') and response.tool_calls:
            assert len(response.tool_calls) > 0
            assert response.tool_calls[0]['name'] == 'get_weather'
        else:
            # If no tool calls, should at least have content
            assert response.content is not None
    
    def test_structured_output_json_mode(self):
        """Test structured output with JSON mode."""
        class PersonInfo(BaseModel):
            name: str
            age: int
            occupation: str
        
        structured_llm = self.llm.with_structured_output(PersonInfo, method="json_mode")
        
        response = structured_llm.invoke(
            "Extract information: John Smith is a 30-year-old software engineer."
        )
        
        assert isinstance(response, PersonInfo)
        assert response.name
        assert response.age > 0
        assert response.occupation
    
    def test_batch_processing(self):
        """Test batch processing functionality."""
        messages_list = [
            [HumanMessage(content="What is 2+2?")],
            [HumanMessage(content="What is 3+3?")],
            [HumanMessage(content="What is 4+4?")],
        ]
        
        responses = self.llm.batch(messages_list)
        
        assert len(responses) == 3
        for response in responses:
            assert response.content is not None
            assert len(response.content) > 0
            assert "reasoning_content" in response.additional_kwargs
    
    @pytest.mark.asyncio
    async def test_async_batch_processing(self):
        """Test async batch processing functionality."""
        messages_list = [
            [HumanMessage(content="Name one benefit of exercise.")],
            [HumanMessage(content="Name one benefit of reading.")],
        ]
        
        responses = await self.llm.abatch(messages_list)
        
        assert len(responses) == 2
        for response in responses:
            assert response.content is not None
            assert len(response.content) > 0
            assert "reasoning_content" in response.additional_kwargs
    
    def test_enable_thinking_parameter(self):
        """Test the enable_thinking parameter functionality."""
        # Test with thinking enabled
        llm_with_thinking = ChatQwenVllm(
            model="Qwen/Qwen3-32B",
            enable_thinking=True,
            temperature=0.1,
        )
        
        response_with_thinking = llm_with_thinking.invoke("Solve: 25 × 4")
        assert "reasoning_content" in response_with_thinking.additional_kwargs
        
        # Test with thinking disabled
        llm_without_thinking = ChatQwenVllm(
            model="Qwen/Qwen3-32B",
            enable_thinking=False,
            temperature=0.1,
        )
        
        response_without_thinking = llm_without_thinking.invoke("Solve: 25 × 4")
        # Should still have the key but content might be empty
        assert "reasoning_content" in response_without_thinking.additional_kwargs
    
    def test_model_configuration(self):
        """Test model configuration and properties."""
        llm = ChatQwenVllm(
            model="Qwen/Qwen3-32B",
            temperature=0.5,
            max_tokens=150,
        )
        
        # Test model properties
        assert llm._llm_type == "chat-qwen"
        assert llm._support_tool_choice() is True
        assert llm._check_need_stream() is False  # VLLM can handle both efficiently
        
        # Test that it can invoke successfully
        response = llm.invoke("Hello")
        assert response.content is not None


class TestChatQwenVllmErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_message_handling(self):
        """Test handling of empty messages."""
        llm = ChatQwenVllm(model="Qwen/Qwen3-32B")
        
        # Test with empty content - should handle gracefully
        try:
            response = llm.invoke("")
            # If it succeeds, should have some response
            assert hasattr(response, 'content')
        except Exception as e:
            # If it fails, should be a meaningful error
            assert isinstance(e, (ValueError, TypeError))
    
    def test_invalid_model_graceful_failure(self):
        """Test graceful handling of invalid model names."""
        # This test might need adjustment based on actual VLLM behavior
        llm = ChatQwenVllm(model="nonexistent-model")
        
        # The actual behavior depends on VLLM server configuration
        # This test ensures our code doesn't crash before making the request
        assert llm.model_name == "nonexistent-model"
        assert llm._llm_type == "chat-qwen"
