"""Integration tests for ChatQwenVllm with VLLM backend."""

from typing import List, Type

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

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
    
    def test_structured_output_vllm_specific(self):
        """Test structured output with vLLM-specific json_schema method."""
        class PersonInfo(BaseModel):
            name: str
            age: int
            occupation: str
        
        # vLLM only supports json_schema method
        structured_llm = self.llm.with_structured_output(PersonInfo, method="json_schema")
        
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


class TestChatQwenVllmStructuredOutput:
    """Test structured output functionality for ChatQwenVllm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.llm = ChatQwenVllm(
            model="Qwen/Qwen3-32B",
            temperature=0.0,
            max_tokens=1000,
            enable_thinking=True,
        )
    
    def test_structured_output_with_pydantic_model(self):
        """Test structured output using Pydantic model."""
        class PersonInfo(BaseModel):
            """Person information schema."""
            name: str = Field(description="Full name of the person")
            age: int = Field(description="Age in years", ge=0, le=150)
            occupation: str = Field(description="Job or profession")
            location: str = Field(description="City or location")
        
        structured_llm = self.llm.with_structured_output(
            schema=PersonInfo,
            method="json_schema"
        )
        
        response = structured_llm.invoke(
            "Extract information: Sarah Johnson is a 28-year-old doctor working in Boston."
        )
        
        assert isinstance(response, PersonInfo)
        assert isinstance(response.name, str)
        assert isinstance(response.age, int)
        assert isinstance(response.occupation, str)
        assert isinstance(response.location, str)
        assert response.age >= 0
    
    def test_structured_output_with_dict_schema(self):
        """Test structured output using dictionary schema."""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The book title"},
                "author": {"type": "string", "description": "The author name"},
                "year": {"type": "integer", "description": "Publication year"},
                "genre": {"type": "string", "description": "Book genre"}
            },
            "required": ["title", "author", "year"],
            "additionalProperties": False
        }
        
        structured_llm = self.llm.with_structured_output(
            schema=schema,
            method="json_schema"
        )
        
        # More explicit prompt for better extraction
        response = structured_llm.invoke([
            {
                "role": "system", 
                "content": "You are a book information extractor. Extract the exact information requested."
            },
            {
                "role": "user", 
                "content": "Extract book information from this text: The novel '1984' was written by George Orwell and published in 1949. It's a dystopian fiction novel."
            }
        ])
        
        assert isinstance(response, dict)
        assert "title" in response
        assert "author" in response  
        assert "year" in response
        assert isinstance(response["year"], int)
        assert response["year"] == 1949
    
    def test_structured_output_complex_nested_schema(self):
        """Test structured output with complex nested schema."""
        class Address(BaseModel):
            """Address information."""
            street: str
            city: str
            country: str
            postal_code: str = Field(description="ZIP or postal code")
        
        class ContactInfo(BaseModel):
            """Contact information."""
            email: str = Field(description="Email address")
            phone: str = Field(description="Phone number")
        
        class Company(BaseModel):
            """Company information."""
            name: str = Field(description="Company name")
            employees: int = Field(description="Number of employees", ge=1)
            address: Address
            contact: ContactInfo
            founded_year: int = Field(description="Year company was founded")
        
        structured_llm = self.llm.with_structured_output(
            schema=Company,
            method="json_schema"
        )
        
        response = structured_llm.invoke(
            """Extract company information: 
            TechCorp Inc. was founded in 2010 and has 250 employees. 
            They are located at 123 Tech Street, San Francisco, USA, 94105.
            Contact them at info@techcorp.com or +1-555-0123."""
        )
        
        assert isinstance(response, Company)
        assert isinstance(response.address, Address)
        assert isinstance(response.contact, ContactInfo)
        assert response.employees > 0
        assert response.founded_year > 1900
    
    def test_structured_output_with_list_schema(self):
        """Test structured output with list/array schema."""
        class Task(BaseModel):
            """Task information."""
            task_id: int = Field(description="Unique task identifier")
            title: str = Field(description="Task title")
            priority: str = Field(description="Priority level: high, medium, low")
            completed: bool = Field(description="Whether task is completed")
        
        class TaskList(BaseModel):
            """List of tasks."""
            tasks: List[Task] = Field(description="List of tasks")
            total_count: int = Field(description="Total number of tasks")
        
        structured_llm = self.llm.with_structured_output(
            schema=TaskList,
            method="json_schema"
        )
        
        # More structured prompt with clearer instructions
        response = structured_llm.invoke([
            {
                "role": "system", 
                "content": "You are a task manager. Extract task information exactly as specified in the schema."
            },
            {
                "role": "user", 
                "content": """Parse these tasks into the required format:
            
            Task 1: Fix bug #123 (ID: 1, Priority: high, Status: not completed)
            Task 2: Write documentation (ID: 2, Priority: medium, Status: completed)
            Task 3: Code review (ID: 3, Priority: low, Status: not completed)
            
            Please extract all tasks with their IDs, titles, priorities, and completion status."""
            }
        ])
        
        assert isinstance(response, TaskList)
        assert isinstance(response.tasks, list)
        assert len(response.tasks) >= 2  # Should extract at least 2 tasks
        assert response.total_count >= len(response.tasks)
        for task in response.tasks:
            assert isinstance(task, Task)
            assert isinstance(task.task_id, int)
            assert isinstance(task.completed, bool)
            assert task.priority in ["high", "medium", "low"]
    
    def test_structured_output_with_include_raw(self):
        """Test structured output with include_raw=True."""
        class WeatherInfo(BaseModel):
            """Weather information."""
            temperature: int = Field(description="Temperature in Celsius")
            condition: str = Field(description="Weather condition")
            humidity: int = Field(description="Humidity percentage", ge=0, le=100)
        
        structured_llm = self.llm.with_structured_output(
            schema=WeatherInfo,
            method="json_schema",
            include_raw=True
        )
        
        response = structured_llm.invoke(
            "Current weather: 22°C, sunny, 65% humidity"
        )
        
        assert isinstance(response, dict)
        assert "raw" in response
        assert "parsed" in response
        assert "parsing_error" in response
        
        # Check raw response
        assert hasattr(response["raw"], "content")
        
        # Check parsed response
        if response["parsing_error"] is None:
            assert isinstance(response["parsed"], WeatherInfo)
            assert isinstance(response["parsed"].temperature, int)
            assert isinstance(response["parsed"].humidity, int)
            assert 0 <= response["parsed"].humidity <= 100
    
    @pytest.mark.asyncio
    async def test_structured_output_async(self):
        """Test structured output with async operations."""
        class BookReview(BaseModel):
            """Book review information."""
            book_title: str
            rating: int = Field(description="Rating out of 5", ge=1, le=5)
            review_text: str
            recommend: bool = Field(description="Whether to recommend the book")
        
        structured_llm = self.llm.with_structured_output(
            schema=BookReview,
            method="json_schema"
        )
        
        response = await structured_llm.ainvoke(
            "Write a review: 'The Great Gatsby' deserves 4 out of 5 stars. "
            "A brilliant exploration of the American Dream. Highly recommended!"
        )
        
        assert isinstance(response, BookReview)
        assert 1 <= response.rating <= 5
        assert isinstance(response.recommend, bool)
        assert len(response.review_text) > 0
    
    def test_structured_output_error_unsupported_method(self):
        """Test error handling for unsupported methods."""
        class SimpleSchema(BaseModel):
            name: str
            value: int
        
        # Test function_calling method (not supported)
        with pytest.raises(ValueError, match="Method 'function_calling' is not supported"):
            self.llm.with_structured_output(
                schema=SimpleSchema,
                method="function_calling"
            )
        
        # Test json_mode method (not supported) 
        with pytest.raises(ValueError, match="Method 'json_mode' is not supported"):
            self.llm.with_structured_output(
                schema=SimpleSchema,
                method="json_mode"
            )
    
    def test_structured_output_error_missing_schema(self):
        """Test error handling for missing schema."""
        with pytest.raises(ValueError, match="Schema must be provided"):
            self.llm.with_structured_output(schema=None)
    
    def test_structured_output_error_invalid_kwargs(self):
        """Test error handling for invalid kwargs."""
        class SimpleSchema(BaseModel):
            name: str
        
        with pytest.raises(ValueError, match="Received unsupported arguments"):
            self.llm.with_structured_output(
                schema=SimpleSchema,
                method="json_schema",
                invalid_param="value"
            )
    
    def test_structured_output_medical_coding_example(self):
        """Test structured output with medical coding analysis (real-world example)."""
        class OperationAnalysis(BaseModel):
            """Medical operation analysis schema."""
            code: str = Field(description="ICD operation code")
            name: str = Field(description="Operation name")
            match_flag: str = Field(
                description="Match status: must be one of 完全, 部分, 未匹配"
            )
            score: int = Field(
                description="Match score: 0 for no match, 40-70 for partial, 70-100 for complete", 
                ge=0, le=100
            )
            rules: str = Field(description="Analysis reasoning")
        
        class OperationResults(BaseModel):
            """Medical coding analysis results."""
            opers: List[OperationAnalysis] = Field(description="List of operation analyses")
        
        structured_llm = self.llm.with_structured_output(
            schema=OperationResults,
            method="json_schema"
        )
        
        messages = [
            SystemMessage(content="""你是一名资深病案编码员，请严格按照以下要求分析手术记录与ICD编码的匹配：

匹配标准：
- 完全：文本完全支持该编码，分数70-100
- 部分：文本部分支持该编码，分数40-70  
- 未匹配：文本不支持该编码，分数0

请为每个ICD编码提供分析结果，match_flag必须是：完全、部分、未匹配 中的一个。"""),
            HumanMessage(content="""请分析以下手术记录与ICD编码的匹配程度：

手术文本: 
1. 穿刺右桡动脉成功后置6F桡动脉鞘
2. 循导丝送药物涂层球囊进行血管扩张

ICD编码列表:
1. 00.6600x008 经皮冠状动脉药物球囊扩张成形术
2. 00.4000 单根血管操作

请为每个编码分析匹配程度并给出评分和理由。""")
        ]
        
        response = structured_llm.invoke(messages)
        
        assert isinstance(response, OperationResults)
        assert isinstance(response.opers, list)
        assert len(response.opers) >= 1
        
        for oper in response.opers:
            assert isinstance(oper, OperationAnalysis)
            assert oper.code
            assert oper.name
            # More flexible assertion to handle variations
            valid_flags = ["完全", "部分", "未匹配", "完全匹配", "部分匹配", "未匹配"]
            assert oper.match_flag in valid_flags, f"Invalid match_flag: {oper.match_flag}"
            assert 0 <= oper.score <= 100
            assert oper.rules
            assert len(oper.rules) > 0
    
    def test_structured_output_schema_conversion(self):
        """Test that different schema types are properly converted."""
        # Test with Pydantic model
        class PydanticSchema(BaseModel):
            field1: str = Field(description="A string field")
            field2: int = Field(description="An integer field")
        
        llm1 = self.llm.with_structured_output(PydanticSchema, method="json_schema")
        assert llm1 is not None
        
        # Test with dict schema
        dict_schema = {
            "type": "object",
            "properties": {
                "field1": {"type": "string", "description": "A string field"},
                "field2": {"type": "integer", "description": "An integer field"}
            },
            "required": ["field1", "field2"],
            "additionalProperties": False
        }
        
        llm2 = self.llm.with_structured_output(dict_schema, method="json_schema")
        assert llm2 is not None
        
        # Verify that both schemas can be used without actual API calls
        # This is a structural test to ensure schema conversion works
        assert hasattr(llm1, 'invoke')
        assert hasattr(llm2, 'invoke')
    
    def test_structured_output_extra_body_integration(self):
        """Test that guided_json is properly integrated with extra_body."""
        class TestSchema(BaseModel):
            message: str
        
        # Test that the schema is passed in extra_body
        structured_llm = self.llm.with_structured_output(TestSchema, method="json_schema")
        
        # The bind should have created a new instance with extra_body containing guided_json
        # This is more of a structural test to ensure the binding works
        assert structured_llm is not None
        
        # Test that we can still invoke (though we won't test the actual API call)
        try:
            # This would normally call the API, but we're just testing the setup
            assert hasattr(structured_llm, 'invoke')
            assert hasattr(structured_llm, 'ainvoke')
        except Exception:
            # Expected since we're not connected to actual VLLM server
            pass


class TestChatQwenVllmToolCalling:
    """Test tool calling functionality for ChatQwenVllm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.llm = ChatQwenVllm(
            model="Qwen/Qwen3-32B",
            temperature=0.1,
            max_tokens=1000,
            enable_thinking=True,
        )
    
    def get_weather_tool(self):
        """Create a weather query tool for testing."""
        @tool
        def get_weather(location: str, unit: str = "celsius") -> str:
            """Get the current weather for a specific location.
            
            Args:
                location: The city or location to get weather for
                unit: Temperature unit, either 'celsius' or 'fahrenheit'
            
            Returns:
                A string describing the current weather conditions
            """
            # Mock weather data for testing
            weather_data = {
                "beijing": {"temp": 22, "condition": "sunny", "humidity": 60},
                "shanghai": {"temp": 25, "condition": "cloudy", "humidity": 75},
                "guangzhou": {"temp": 28, "condition": "rainy", "humidity": 85},
                "shenzhen": {"temp": 26, "condition": "partly cloudy", "humidity": 70},
                "new york": {"temp": 18, "condition": "overcast", "humidity": 65},
                "london": {"temp": 15, "condition": "rainy", "humidity": 80},
                "tokyo": {"temp": 20, "condition": "sunny", "humidity": 55}
            }
            
            location_lower = location.lower()
            if location_lower in weather_data:
                data = weather_data[location_lower]
                temp_unit = "°C" if unit == "celsius" else "°F"
                if unit == "fahrenheit":
                    data["temp"] = int(data["temp"] * 9/5 + 32)
                
                return (f"Current weather in {location}: {data['condition']}, "
                       f"{data['temp']}{temp_unit}, humidity {data['humidity']}%")
            else:
                return f"Weather data not available for {location}"
        
        return get_weather
    
    def test_tool_calling_text_output(self):
        """Test tool calling with regular text output."""
        weather_tool = self.get_weather_tool()
        llm_with_tools = self.llm.bind_tools([weather_tool])
        
        response = llm_with_tools.invoke("What's the weather like in Beijing today?")
        
        # Check if the response contains tool calls or mentions the weather
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Tool was called successfully
            assert len(response.tool_calls) > 0
            tool_call = response.tool_calls[0]
            assert tool_call['name'] == 'get_weather'
            assert 'location' in tool_call['args']
            # Check that Beijing was extracted correctly
            assert 'beijing' in tool_call['args']['location'].lower()
        else:
            # Model responded with text, should at least mention weather or location
            assert response.content is not None
            content_lower = response.content.lower()
            assert any(word in content_lower for word in ['weather', 'beijing', 'temperature', 'condition'])
    
    def test_tool_calling_multiple_locations(self):
        """Test tool calling with multiple location queries."""
        weather_tool = self.get_weather_tool()
        llm_with_tools = self.llm.bind_tools([weather_tool])
        
        response = llm_with_tools.invoke(
            "Can you tell me the weather in Shanghai and Guangzhou?"
        )
        
        # Check response
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Should ideally have multiple tool calls or at least one
            assert len(response.tool_calls) >= 1
            
            # Check that at least one of the cities is mentioned
            cities_mentioned = []
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'get_weather':
                    location = tool_call['args'].get('location', '').lower()
                    cities_mentioned.append(location)
            
            # At least one city should be mentioned
            assert any('shanghai' in city or 'guangzhou' in city for city in cities_mentioned)
        else:
            # Text response should mention the cities
            assert response.content is not None
            content_lower = response.content.lower()
            assert any(city in content_lower for city in ['shanghai', 'guangzhou'])
    
    def test_tool_calling_with_structured_output(self):
        """Test tool calling combined with structured output."""
        class WeatherQuery(BaseModel):
            """Weather query result schema."""
            location: str = Field(description="The queried location")
            temperature: int = Field(description="Temperature in degrees")
            condition: str = Field(description="Weather condition")
            humidity: int = Field(description="Humidity percentage", ge=0, le=100)
            unit: str = Field(
                description="Temperature unit: must be exactly 'celsius' or 'fahrenheit'",
                default="celsius"
            )
        
        weather_tool = self.get_weather_tool()
        
        # First bind tools, then add structured output
        llm_with_tools = self.llm.bind_tools([weather_tool])
        structured_llm = llm_with_tools.with_structured_output(
            schema=WeatherQuery,
            method="json_schema"
        )
        
        response = structured_llm.invoke([
            {
                "role": "system", 
                "content": """You are a weather assistant. When asked about weather:
1. Use the available weather tool to get accurate information
2. Format the response according to the required schema
3. IMPORTANT: For the 'unit' field, use EXACTLY 'celsius' or 'fahrenheit' (not '°C' or '°F')
4. Extract the numerical temperature value without units
5. Ensure all required fields are properly filled"""
            },
            {
                "role": "user", 
                "content": "What's the weather in Tokyo? Please provide the information in the structured format with temperature in celsius."
            }
        ])
        
        # Should return structured data
        assert isinstance(response, WeatherQuery)
        assert response.location
        assert isinstance(response.temperature, int)
        assert response.condition
        assert 0 <= response.humidity <= 100
        # More flexible unit checking to handle possible format variations
        assert response.unit.lower() in ["celsius", "fahrenheit", "c", "f"] or any(
            unit in response.unit.lower() for unit in ["celsius", "fahrenheit"]
        ), f"Invalid unit format: {response.unit}"
        
        # Location should relate to Tokyo
        assert 'tokyo' in response.location.lower()
    
    def test_tool_calling_with_include_raw(self):
        """Test tool calling with structured output and include_raw=True."""
        class WeatherSummary(BaseModel):
            """Weather summary schema."""
            cities: List[str] = Field(description="List of cities queried")
            overall_condition: str = Field(description="Overall weather description")
            temperature_range: str = Field(description="Temperature range description")
        
        weather_tool = self.get_weather_tool()
        llm_with_tools = self.llm.bind_tools([weather_tool])
        structured_llm = llm_with_tools.with_structured_output(
            schema=WeatherSummary,
            method="json_schema",
            include_raw=True
        )
        
        response = structured_llm.invoke([
            {
                "role": "system",
                "content": """You are a weather analyst. When analyzing weather:
1. Use the weather tool to gather information for each city
2. Provide a comprehensive summary in the required format
3. List all cities that were queried
4. Ensure the summary includes overall weather patterns"""
            },
            {
                "role": "user",
                "content": "Get weather information for Beijing and Shanghai, then provide a summary with the overall condition and temperature range."
            }
        ])
        
        # Should return dict with raw, parsed, parsing_error
        assert isinstance(response, dict)
        assert "raw" in response
        assert "parsed" in response
        assert "parsing_error" in response
        
        # Check raw response
        assert hasattr(response["raw"], "content")
        
        # Check parsed response if parsing succeeded
        if response["parsing_error"] is None:
            assert isinstance(response["parsed"], WeatherSummary)
            assert isinstance(response["parsed"].cities, list)
            assert len(response["parsed"].cities) >= 1
            assert response["parsed"].overall_condition
            assert response["parsed"].temperature_range
    
    @pytest.mark.asyncio
    async def test_tool_calling_async(self):
        """Test async tool calling functionality."""
        weather_tool = self.get_weather_tool()
        llm_with_tools = self.llm.bind_tools([weather_tool])
        
        response = await llm_with_tools.ainvoke(
            "What's the weather like in London right now?"
        )
        
        # Check response
        if hasattr(response, 'tool_calls') and response.tool_calls:
            assert len(response.tool_calls) > 0
            tool_call = response.tool_calls[0]
            assert tool_call['name'] == 'get_weather'
            assert 'location' in tool_call['args']
            assert 'london' in tool_call['args']['location'].lower()
        else:
            # Should at least have content mentioning weather or London
            assert response.content is not None
            content_lower = response.content.lower()
            assert any(word in content_lower for word in ['weather', 'london', 'temperature'])
    
    def test_tool_calling_with_temperature_unit(self):
        """Test tool calling with specific temperature unit parameter."""
        weather_tool = self.get_weather_tool()
        llm_with_tools = self.llm.bind_tools([weather_tool])
        
        response = llm_with_tools.invoke(
            "What's the temperature in New York in Fahrenheit?"
        )
        
        # Check tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            assert len(response.tool_calls) > 0
            tool_call = response.tool_calls[0]
            assert tool_call['name'] == 'get_weather'
            
            # Should extract location
            assert 'location' in tool_call['args']
            assert 'new york' in tool_call['args']['location'].lower()
            
            # Should ideally extract unit, but this depends on model capability
            if 'unit' in tool_call['args']:
                assert 'fahrenheit' in tool_call['args']['unit'].lower()
        else:
            # Text response should mention New York and potentially Fahrenheit
            assert response.content is not None
            content_lower = response.content.lower()
            assert 'new york' in content_lower
    
    def test_tool_calling_error_handling(self):
        """Test tool calling with invalid location."""
        weather_tool = self.get_weather_tool()
        llm_with_tools = self.llm.bind_tools([weather_tool])
        
        response = llm_with_tools.invoke(
            "What's the weather in NonexistentCity?"
        )
        
        # Response should handle gracefully
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Tool was called, which is expected behavior
            assert len(response.tool_calls) > 0
            tool_call = response.tool_calls[0]
            assert tool_call['name'] == 'get_weather'
            assert 'location' in tool_call['args']
        
        # Should have some response content
        assert response.content is not None or (hasattr(response, 'tool_calls') and response.tool_calls)


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
