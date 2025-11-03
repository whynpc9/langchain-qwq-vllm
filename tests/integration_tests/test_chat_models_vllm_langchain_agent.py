"""Integration tests for ChatQwenVllm with LangChain 1.x agents."""

import os
from typing import Literal

import pytest
from dotenv import load_dotenv

from langchain_qwq.chat_models_vllm import ChatQwenVllm
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()


class TestChatQwenVllmWithLangChainAgent:
    """Test ChatQwenVllm integration with LangChain 1.x agents."""

    def setup_method(self):
        """Set up test fixtures."""
        # Set dummy API key for VLLM (doesn't need real OpenAI key)
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "dummy-key-for-vllm"
        
        self.llm = ChatQwenVllm(
            model="Qwen/Qwen3-32B",
            temperature=0.1,
            max_tokens=2000,
            enable_thinking=True,
        )

    def test_basic_agent_creation(self):
        """Test basic agent creation with ChatQwenVllm."""
        
        @tool
        def simple_calculator(operation: str, a: float, b: float) -> str:
            """Perform simple math operations.
            
            Args:
                operation: The operation to perform (add, subtract, multiply, divide)
                a: First number
                b: Second number
                
            Returns:
                Result of the operation as string
            """
            if operation == "add":
                return f"{a} + {b} = {a + b}"
            elif operation == "subtract":
                return f"{a} - {b} = {a - b}"
            elif operation == "multiply":
                return f"{a} * {b} = {a * b}"
            elif operation == "divide":
                if b != 0:
                    return f"{a} / {b} = {a / b}"
                else:
                    return "Error: Cannot divide by zero"
            else:
                return f"Error: Unknown operation '{operation}'"

        # Create the agent with ChatQwenVllm
        agent = create_agent(
            model=self.llm,
            tools=[simple_calculator],
            system_prompt=(
                "You are an expert mathematician and problem solver. "
                "Your job is to solve mathematical problems step by step. "
                "You have access to a calculator tool for basic operations."
            )
        )

        # Test the agent (requires VLLM server running)
        # Note: Actual invocation will fail without server, but creation should succeed
        assert agent is not None
        print(f"Agent created successfully: {type(agent)}")

    def test_agent_with_calculator(self):
        """Test agent with calculator tool execution."""
        
        @tool
        def calculator(operation: str, a: float, b: float) -> str:
            """Perform basic math operations."""
            operations = {
                "add": lambda x, y: x + y,
                "subtract": lambda x, y: x - y,
                "multiply": lambda x, y: x * y,
                "divide": lambda x, y: x / y if y != 0 else "Error: division by zero"
            }
            if operation in operations:
                result = operations[operation](a, b)
                return f"{a} {operation} {b} = {result}"
            return f"Error: unknown operation {operation}"

        agent = create_agent(
            model=self.llm,
            tools=[calculator],
            system_prompt="You are a helpful math assistant."
        )

        # Test with a simple calculation
        result = agent.invoke({
            "messages": [
                HumanMessage(content="Calculate 15 * 8 + 23")
            ]
        })

        # Verify we got a result
        assert "messages" in result
        assert len(result["messages"]) > 0
        
        final_message = result["messages"][-1]
        assert hasattr(final_message, 'content')
        assert final_message.content is not None
        
        print(f"Final answer: {final_message.content}")

    def test_agent_with_multiple_tools(self):
        """Test agent with multiple tools."""
        
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b
        
        @tool
        def power(base: int, exponent: int) -> int:
            """Calculate base raised to exponent."""
            return base ** exponent

        agent = create_agent(
            model=self.llm,
            tools=[add, multiply, power],
            system_prompt="You are a math expert. Use the tools to solve problems step by step."
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(content="Calculate (2 + 3) * 4 and then raise that to power 2")
            ]
        })

        assert "messages" in result
        assert len(result["messages"]) > 0

    def test_agent_with_thinking_enabled(self):
        """Test that thinking mode works with agents."""
        
        @tool
        def analyze_number(n: int) -> str:
            """Analyze properties of a number."""
            is_even = n % 2 == 0
            is_prime = n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))
            return f"Number {n}: even={is_even}, prime={is_prime}"

        # Ensure thinking is enabled
        llm_thinking = ChatQwenVllm(
            model="Qwen/Qwen3-32B",
            temperature=0.1,
            max_tokens=2000,
            enable_thinking=True,
        )

        agent = create_agent(
            model=llm_thinking,
            tools=[analyze_number],
            system_prompt="You are a number theory expert."
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(content="Analyze the number 17")
            ]
        })

        assert "messages" in result
        # Check if we got reasoning in the response
        for msg in result["messages"]:
            if hasattr(msg, 'additional_kwargs'):
                if "reasoning_content" in msg.additional_kwargs:
                    print(f"Found reasoning: {msg.additional_kwargs['reasoning_content'][:100]}...")
                    break

    def test_agent_with_error_handling(self):
        """Test agent error handling with failing tools."""
        
        @tool
        def unreliable_tool(operation: str) -> str:
            """A tool that sometimes fails."""
            if operation == "fail":
                raise ValueError("Simulated tool failure")
            return f"Operation '{operation}' completed successfully"

        agent = create_agent(
            model=self.llm,
            tools=[unreliable_tool],
            system_prompt="You are a resilient assistant that handles tool failures gracefully."
        )

        # This should succeed (agent creation)
        assert agent is not None
        print("Agent with error-prone tool created successfully")

    def test_streaming_agent_execution(self):
        """Test agent execution with streaming."""
        
        @tool
        def get_info(topic: str) -> str:
            """Get information about a topic."""
            return f"Information about {topic}: This is a test response."

        agent = create_agent(
            model=self.llm,
            tools=[get_info],
            system_prompt="You are an informative assistant."
        )

        # Stream the execution
        chunks = []
        for chunk in agent.stream({
            "messages": [
                HumanMessage(content="Tell me about Python programming")
            ]
        }, stream_mode="updates"):
            chunks.append(chunk)
            print(f"Received chunk: {chunk}")

        assert len(chunks) > 0

    def test_agent_compatibility_with_enable_thinking(self):
        """Test that enable_thinking parameter is properly set."""
        llm = ChatQwenVllm(
            model="Qwen/Qwen3-32B",
            enable_thinking=True,
        )
        
        # Check that enable_thinking is set
        assert llm.enable_thinking is True
        
        # Access _default_params to trigger extra_body initialization
        params = llm._default_params
        
        # Now check extra_body configuration
        assert llm.extra_body is not None
        assert "chat_template_kwargs" in llm.extra_body
        assert llm.extra_body["chat_template_kwargs"]["enable_thinking"] is True
        
        print(f"✓ enable_thinking properly configured: {llm.extra_body}")

    def test_tool_binding_compatibility(self):
        """Test that tool binding works with LangChain 1.x."""
        
        @tool
        def test_tool(query: str) -> str:
            """A test tool."""
            return f"Processed: {query}"
        
        llm_with_tools = self.llm.bind_tools([test_tool])
        
        # Verify it's a RunnableBinding
        from langchain_core.runnables.base import RunnableBinding
        assert isinstance(llm_with_tools, RunnableBinding)
        print("✓ Tool binding works correctly")


class TestChatQwenVllmCompatibility:
    """Test compatibility requirements for LangChain 1.x."""
    
    def test_required_methods_exist(self):
        """Test that ChatQwenVllm has required methods for LangChain 1.x."""
        # Set dummy API key
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "dummy-key-for-vllm"
        
        llm = ChatQwenVllm(model="Qwen/Qwen3-32B")
        
        # Check required methods
        assert hasattr(llm, 'invoke')
        assert hasattr(llm, 'ainvoke')
        assert hasattr(llm, 'stream')
        assert hasattr(llm, 'astream')
        assert hasattr(llm, 'bind_tools')
        
        # Check VLLM-specific features
        assert hasattr(llm, 'enable_thinking')
        assert llm._support_tool_choice() is True
        
        print("✓ All required methods present")
    
    def test_langchain_version(self):
        """Test that we're using LangChain 1.x."""
        import langchain
        import langchain_core
        
        # Check versions
        assert langchain.__version__.startswith("1."), f"Expected LangChain 1.x, got {langchain.__version__}"
        assert langchain_core.__version__.startswith("1."), f"Expected langchain-core 1.x, got {langchain_core.__version__}"
        
        print(f"✓ LangChain versions:")
        print(f"  - langchain: {langchain.__version__}")
        print(f"  - langchain-core: {langchain_core.__version__}")
    
    def test_create_agent_available(self):
        """Test that create_agent is available from langchain.agents."""
        from langchain.agents import create_agent
        
        assert create_agent is not None
        print("✓ create_agent available from langchain.agents")


if __name__ == "__main__":
    # Run basic compatibility tests
    print("Running LangChain 1.x Compatibility Tests")
    print("=" * 60)
    
    test_compat = TestChatQwenVllmCompatibility()
    test_compat.test_langchain_version()
    test_compat.test_create_agent_available()
    test_compat.test_required_methods_exist()
    
    print("\n" + "=" * 60)
    print("Running Agent Tests")
    print("=" * 60)
    
    test_agent = TestChatQwenVllmWithLangChainAgent()
    test_agent.setup_method()
    test_agent.test_basic_agent_creation()
    test_agent.test_agent_with_error_handling()
    test_agent.test_agent_compatibility_with_enable_thinking()
    test_agent.test_tool_binding_compatibility()
    
    print("\n" + "=" * 60)
    print("✓ All non-server tests passed!")
    print("Note: Tests requiring VLLM server are skipped")
    print("=" * 60)

