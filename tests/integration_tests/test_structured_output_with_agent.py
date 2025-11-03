"""Integration tests for structured output with LangChain 1.x agents."""

import os
from typing import Literal

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_qwq.chat_models_vllm import ChatQwenVllm
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import HumanMessage

load_dotenv()


class TestStructuredOutputWithAgent:
    """Test structured output integration with LangChain 1.x agents."""

    def setup_method(self):
        """Set up test fixtures."""
        # Set dummy API key for VLLM
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "dummy-key-for-vllm"
        
        self.llm = ChatQwenVllm(
            model="Qwen/Qwen3-32B",
            temperature=0.1,
            max_tokens=2000,
            enable_thinking=False,  # Disable thinking for structured output
        )

    def test_simple_structured_output_with_pydantic(self):
        """Test simple structured output using Pydantic model."""
        
        class ContactInfo(BaseModel):
            """Contact information for a person."""
            name: str = Field(description="The name of the person")
            email: str = Field(description="The email address of the person")
            phone: str = Field(description="The phone number of the person")

        agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt="You are a helpful assistant that extracts contact information.",
            response_format=ProviderStrategy(ContactInfo)
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(content="Extract contact info from: John Doe, john@example.com, (555) 123-4567")
            ]
        })

        # Check that structured_response exists and is correctly formatted
        assert "structured_response" in result
        structured_data = result["structured_response"]
        
        assert isinstance(structured_data, ContactInfo)
        assert structured_data.name == "John Doe"
        assert structured_data.email == "john@example.com"
        assert structured_data.phone == "(555) 123-4567"
        
        print(f"✓ Structured output: {structured_data}")

    def test_structured_output_with_complex_schema(self):
        """Test structured output with nested fields."""
        
        class Address(BaseModel):
            """Physical address."""
            street: str = Field(description="Street address")
            city: str = Field(description="City name")
            state: str = Field(description="State or province")
            zip_code: str = Field(description="Postal code")

        class Person(BaseModel):
            """Person with address."""
            name: str = Field(description="Full name")
            age: int = Field(description="Age in years")
            address: Address = Field(description="Physical address")

        agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt="You are a helpful assistant that extracts person information.",
            response_format=ProviderStrategy(Person)
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(content="Extract info: Alice Smith, 30 years old, lives at 123 Main St, San Francisco, CA 94102")
            ]
        })

        assert "structured_response" in result
        structured_data = result["structured_response"]
        
        assert isinstance(structured_data, Person)
        assert structured_data.name == "Alice Smith"
        assert structured_data.age == 30
        assert structured_data.address.city == "San Francisco"
        
        print(f"✓ Complex structured output: {structured_data}")

    def test_structured_output_with_lists(self):
        """Test structured output with list fields."""
        
        class ProductReview(BaseModel):
            """Analysis of a product review."""
            rating: int = Field(description="Rating from 1-5", ge=1, le=5)
            sentiment: Literal["positive", "negative", "neutral"] = Field(description="Overall sentiment")
            key_points: list[str] = Field(description="Key points mentioned. Lowercase, 1-3 words each.")
            pros: list[str] = Field(description="Positive aspects mentioned")
            cons: list[str] = Field(description="Negative aspects mentioned")

        agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt="You are a helpful assistant that analyzes product reviews.",
            response_format=ProviderStrategy(ProductReview)
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(content="Analyze: 'Great phone! 5 stars. Beautiful display and fast performance. Battery life could be better and it's quite expensive.'")
            ]
        })

        assert "structured_response" in result
        structured_data = result["structured_response"]
        
        assert isinstance(structured_data, ProductReview)
        assert structured_data.rating == 5
        assert structured_data.sentiment in ["positive", "negative", "neutral"]
        assert len(structured_data.pros) > 0
        assert len(structured_data.cons) > 0
        
        print(f"✓ List-based structured output: {structured_data}")

    def test_structured_output_with_optional_fields(self):
        """Test structured output with optional fields."""
        
        class EventInfo(BaseModel):
            """Event information."""
            event_name: str = Field(description="Name of the event")
            date: str = Field(description="Event date")
            location: str | None = Field(default=None, description="Event location if mentioned")
            attendees: int | None = Field(default=None, description="Number of attendees if mentioned")

        agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt="You are a helpful assistant that extracts event information.",
            response_format=ProviderStrategy(EventInfo)
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(content="Extract event info: Tech Conference on March 15th, 2024")
            ]
        })

        assert "structured_response" in result
        structured_data = result["structured_response"]
        
        assert isinstance(structured_data, EventInfo)
        assert "Tech Conference" in structured_data.event_name or "conference" in structured_data.event_name.lower()
        # Date can be in various formats (March 15, 2024-03-15, etc.)
        assert "2024" in structured_data.date or "March" in structured_data.date or "15" in structured_data.date
        # location and attendees should be None as not mentioned
        
        print(f"✓ Optional fields structured output: {structured_data}")

    def test_structured_output_with_enum(self):
        """Test structured output with enum fields."""
        
        class Priority(str):
            """Priority levels."""
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"
            URGENT = "urgent"

        class Task(BaseModel):
            """Task information."""
            title: str = Field(description="Task title")
            description: str = Field(description="Task description")
            priority: Literal["low", "medium", "high", "urgent"] = Field(description="Task priority")
            estimated_hours: int = Field(description="Estimated hours to complete", ge=1, le=100)

        agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt="You are a helpful assistant that creates task information.",
            response_format=ProviderStrategy(Task)
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(content="Create a task: Fix critical database bug - This is blocking production, needs immediate attention, estimated 4 hours")
            ]
        })

        assert "structured_response" in result
        structured_data = result["structured_response"]
        
        assert isinstance(structured_data, Task)
        assert structured_data.priority in ["low", "medium", "high", "urgent"]
        assert structured_data.estimated_hours == 4
        
        print(f"✓ Enum-based structured output: {structured_data}")

    def test_structured_output_with_validation(self):
        """Test that Pydantic validation works correctly."""
        
        class OrderInfo(BaseModel):
            """Order information with validation."""
            order_id: str = Field(description="Order ID", min_length=5, max_length=20)
            quantity: int = Field(description="Quantity ordered", ge=1, le=1000)
            total_price: float = Field(description="Total price", gt=0)
            currency: Literal["USD", "EUR", "GBP"] = Field(description="Currency code")

        agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt="You are a helpful assistant that extracts order information.",
            response_format=ProviderStrategy(OrderInfo)
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(content="Extract order: Order #ORD12345, 5 items, total $99.99 USD")
            ]
        })

        assert "structured_response" in result
        structured_data = result["structured_response"]
        
        assert isinstance(structured_data, OrderInfo)
        assert len(structured_data.order_id) >= 5
        assert structured_data.quantity >= 1
        assert structured_data.total_price > 0
        assert structured_data.currency in ["USD", "EUR", "GBP"]
        
        print(f"✓ Validated structured output: {structured_data}")

    def test_structured_output_accuracy(self):
        """Test accuracy of structured output extraction."""
        
        class MovieInfo(BaseModel):
            """Movie information."""
            title: str = Field(description="Movie title")
            year: int = Field(description="Release year")
            director: str = Field(description="Director name")
            genre: list[str] = Field(description="Movie genres")
            rating: float = Field(description="Rating out of 10", ge=0, le=10)

        agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt="You are a helpful assistant that extracts movie information.",
            response_format=ProviderStrategy(MovieInfo)
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(content="Extract: The Shawshank Redemption (1994), directed by Frank Darabont, is a drama film rated 9.3/10")
            ]
        })

        assert "structured_response" in result
        structured_data = result["structured_response"]
        
        assert isinstance(structured_data, MovieInfo)
        assert "Shawshank" in structured_data.title
        assert structured_data.year == 1994
        assert "Darabont" in structured_data.director
        assert structured_data.rating >= 9.0
        
        print(f"✓ Accurate extraction: {structured_data}")


class TestStructuredOutputCompatibility:
    """Test compatibility of structured output with LangChain 1.x."""

    def test_with_structured_output_still_works(self):
        """Test that the old with_structured_output method still works."""
        # Set dummy API key
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "dummy-key-for-vllm"
        
        class SimpleData(BaseModel):
            name: str = Field(description="Name field")
            value: int = Field(description="Value field")

        llm = ChatQwenVllm(model="Qwen/Qwen3-32B")
        
        # This should still work
        structured_llm = llm.with_structured_output(
            schema=SimpleData,
            method="json_schema"
        )
        
        assert structured_llm is not None
        print("✓ with_structured_output method still functional")


if __name__ == "__main__":
    # Run tests
    print("Running Structured Output Tests")
    print("=" * 60)
    
    test_suite = TestStructuredOutputWithAgent()
    test_suite.setup_method()
    
    print("\nTest 1: Simple Pydantic Model")
    test_suite.test_simple_structured_output_with_pydantic()
    
    print("\nTest 2: Complex Nested Schema")
    test_suite.test_structured_output_with_complex_schema()
    
    print("\nTest 3: Lists and Arrays")
    test_suite.test_structured_output_with_lists()
    
    print("\nTest 4: Optional Fields")
    test_suite.test_structured_output_with_optional_fields()
    
    print("\nTest 5: Enums and Literals")
    test_suite.test_structured_output_with_enum()
    
    print("\nTest 6: Validation")
    test_suite.test_structured_output_with_validation()
    
    print("\nTest 7: Extraction Accuracy")
    test_suite.test_structured_output_accuracy()
    
    print("\n" + "=" * 60)
    print("✓ All structured output tests passed!")
    print("=" * 60)

