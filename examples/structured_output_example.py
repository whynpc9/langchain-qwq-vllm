"""Example of using structured output with ChatQwenVllm and LangChain 1.0 agents.

This example demonstrates how to extract structured data from text using
Pydantic models and VLLM's native guided_json support.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import HumanMessage
from langchain_qwq.chat_models_vllm import ChatQwenVllm

# Load environment variables
load_dotenv()

# Set VLLM API base and dummy OpenAI key
if "VLLM_API_BASE" not in os.environ:
    os.environ["VLLM_API_BASE"] = "http://localhost:8000/v1"
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-vllm"


def example_1_simple_extraction():
    """Example 1: Simple contact information extraction."""
    print("\n" + "=" * 70)
    print("Example 1: Simple Contact Information Extraction")
    print("=" * 70)
    
    class ContactInfo(BaseModel):
        """Contact information for a person."""
        name: str = Field(description="Full name of the person")
        email: str = Field(description="Email address")
        phone: str = Field(description="Phone number")
    
    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        temperature=0.1,
        enable_thinking=False,  # Disable thinking for structured output
    )
    
    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="You are a helpful assistant that extracts contact information.",
        response_format=ProviderStrategy(ContactInfo)
    )
    
    result = agent.invoke({
        "messages": [
            HumanMessage(content="Extract: John Doe, john.doe@example.com, (555) 123-4567")
        ]
    })
    
    contact = result["structured_response"]
    print(f"\nExtracted Contact Information:")
    print(f"  Name:  {contact.name}")
    print(f"  Email: {contact.email}")
    print(f"  Phone: {contact.phone}")


def example_2_nested_structures():
    """Example 2: Nested structures with validation."""
    print("\n" + "=" * 70)
    print("Example 2: Nested Structures with Validation")
    print("=" * 70)
    
    class Address(BaseModel):
        """Physical address."""
        street: str = Field(description="Street address")
        city: str = Field(description="City name")
        state: str = Field(description="State abbreviation", max_length=2)
        zip_code: str = Field(description="ZIP code")
    
    class Person(BaseModel):
        """Person with contact details."""
        name: str = Field(description="Full name")
        age: int = Field(description="Age in years", ge=0, le=150)
        address: Address = Field(description="Physical address")
    
    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        temperature=0.1,
        enable_thinking=False,
    )
    
    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="You are a helpful assistant that extracts person information.",
        response_format=ProviderStrategy(Person)
    )
    
    result = agent.invoke({
        "messages": [
            HumanMessage(content="Alice Smith, 30 years old, lives at 123 Main Street, San Francisco, CA 94102")
        ]
    })
    
    person = result["structured_response"]
    print(f"\nExtracted Person Information:")
    print(f"  Name: {person.name}")
    print(f"  Age:  {person.age}")
    print(f"  Address:")
    print(f"    Street:   {person.address.street}")
    print(f"    City:     {person.address.city}")
    print(f"    State:    {person.address.state}")
    print(f"    ZIP Code: {person.address.zip_code}")


def example_3_complex_analysis():
    """Example 3: Complex data analysis with lists and enums."""
    print("\n" + "=" * 70)
    print("Example 3: Product Review Analysis")
    print("=" * 70)
    
    class ProductReview(BaseModel):
        """Analysis of a product review."""
        rating: int = Field(description="Rating from 1-5", ge=1, le=5)
        sentiment: Literal["positive", "negative", "neutral"] = Field(
            description="Overall sentiment"
        )
        pros: list[str] = Field(description="Positive aspects mentioned")
        cons: list[str] = Field(description="Negative aspects mentioned")
        recommended: bool = Field(description="Whether the product is recommended")
    
    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        temperature=0.1,
        enable_thinking=False,
    )
    
    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="You are a helpful assistant that analyzes product reviews.",
        response_format=ProviderStrategy(ProductReview)
    )
    
    review_text = """
    This laptop is amazing! I give it 5 stars. The display is beautiful and 
    super bright, the performance is lightning fast, and the build quality 
    is excellent. However, the battery life could be better - I only get 
    about 4 hours. Also, it's quite expensive at $2000. Despite these issues, 
    I would definitely recommend it to anyone who needs a powerful machine.
    """
    
    result = agent.invoke({
        "messages": [HumanMessage(content=f"Analyze this review: {review_text}")]
    })
    
    analysis = result["structured_response"]
    print(f"\nReview Analysis:")
    print(f"  Rating:      {analysis.rating}/5")
    print(f"  Sentiment:   {analysis.sentiment}")
    print(f"  Recommended: {analysis.recommended}")
    print(f"\n  Pros:")
    for pro in analysis.pros:
        print(f"    • {pro}")
    print(f"\n  Cons:")
    for con in analysis.cons:
        print(f"    • {con}")


def example_4_optional_fields():
    """Example 4: Optional fields and flexible schemas."""
    print("\n" + "=" * 70)
    print("Example 4: Event Information with Optional Fields")
    print("=" * 70)
    
    class EventInfo(BaseModel):
        """Event information."""
        event_name: str = Field(description="Name of the event")
        date: str = Field(description="Event date")
        location: str | None = Field(default=None, description="Event location if mentioned")
        attendees: int | None = Field(default=None, description="Expected attendees if mentioned")
        virtual: bool = Field(default=False, description="Whether it's a virtual event")
    
    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        temperature=0.1,
        enable_thinking=False,
    )
    
    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="You are a helpful assistant that extracts event information.",
        response_format=ProviderStrategy(EventInfo)
    )
    
    # Example with partial information
    result = agent.invoke({
        "messages": [
            HumanMessage(content="AI Conference 2024 on December 15th")
        ]
    })
    
    event = result["structured_response"]
    print(f"\nExtracted Event Information:")
    print(f"  Event:     {event.event_name}")
    print(f"  Date:      {event.date}")
    print(f"  Location:  {event.location or 'Not specified'}")
    print(f"  Attendees: {event.attendees or 'Not specified'}")
    print(f"  Virtual:   {event.virtual}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ChatQwenVllm Structured Output Examples")
    print("=" * 70)
    print("\nThese examples demonstrate various structured output capabilities:")
    print("  1. Simple field extraction")
    print("  2. Nested structures with validation")
    print("  3. Complex analysis with lists and enums")
    print("  4. Optional fields and flexible schemas")
    
    try:
        example_1_simple_extraction()
        example_2_nested_structures()
        example_3_complex_analysis()
        example_4_optional_fields()
        
        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()

