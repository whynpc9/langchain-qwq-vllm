"""Compare AutoStrategy vs explicit ProviderStrategy."""

import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_qwq.chat_models_vllm import ChatQwenVllm
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import HumanMessage

load_dotenv()

# Set dummy API key for VLLM
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-vllm"

class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

llm = ChatQwenVllm(
    model="Qwen/Qwen3-32B",
    temperature=0.1,
    max_tokens=2000,
    enable_thinking=False,
)

print("=" * 60)
print("Test 1: Using explicit ProviderStrategy (should work)")
print("=" * 60)

try:
    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="You are a helpful assistant that extracts contact information.",
        response_format=ProviderStrategy(ContactInfo)
    )
    
    result = agent.invoke({
        "messages": [
            HumanMessage(content="Extract contact info from: John Doe, john@example.com, (555) 123-4567")
        ]
    })
    
    print(f"✓ Success with ProviderStrategy!")
    print(f"Structured response: {result['structured_response']}\n")
    
except Exception as e:
    print(f"✗ Failed with ProviderStrategy: {e}\n")

print("=" * 60)
print("Test 2: Using direct schema (AutoStrategy)")
print("=" * 60)

try:
    # Create a fresh LLM instance
    llm2 = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        temperature=0.1,
        max_tokens=2000,
        enable_thinking=False,
    )
    
    agent = create_agent(
        model=llm2,
        tools=[],
        system_prompt="You are a helpful assistant that extracts contact information.",
        response_format=ContactInfo  # Auto-selects strategy
    )
    
    result = agent.invoke({
        "messages": [
            HumanMessage(content="Extract contact info from: John Doe, john@example.com, (555) 123-4567")
        ]
    })
    
    print(f"✓ Success with AutoStrategy!")
    print(f"Structured response: {result['structured_response']}\n")
    
except Exception as e:
    print(f"✗ Failed with AutoStrategy: {e}")
    import traceback
    traceback.print_exc()

