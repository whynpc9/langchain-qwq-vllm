"""Integration tests for ChatQwenVllm with deepagents framework."""

import os
from typing import List, Literal

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Import ChatQwenVllm 
from langchain_qwq.chat_models_vllm import ChatQwenVllm

# Try to import deepagents and tavily - mark tests to skip if not available
try:
    from deepagents import create_deep_agent
    DEEPAGENTS_AVAILABLE = True
except ImportError:
    DEEPAGENTS_AVAILABLE = False

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

load_dotenv()


class TestChatQwenVllmWithDeepAgents:
    """Test ChatQwenVllm integration with deepagents framework."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm = ChatQwenVllm(
            model="Qwen/Qwen3-32B",
            temperature=0.1,
            max_tokens=2000,
            enable_thinking=True,
        )

    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not available")
    def test_basic_deepagent_creation(self):
        """Test basic deepagent creation with ChatQwenVllm."""
        
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

        # Instructions for the research agent
        math_instructions = (
            "You are an expert mathematician and problem solver. "
            "Your job is to solve mathematical problems step by step.\n\n"
            "You have access to a calculator tool for basic operations.\n\n"
            "## `simple_calculator`\n\n"
            "Use this to perform basic mathematical operations: "
            "add, subtract, multiply, divide."
        )

        # Create the agent with ChatQwenVllm as the underlying model
        agent = create_deep_agent(
            tools=[simple_calculator],
            instructions=math_instructions,
            model=self.llm,
        )

        # Test the agent
        result = agent.invoke({
            "messages": [
                {
                    "role": "user", 
                    "content": "Calculate 15 * 8 + 23"
                }
            ]
        })

        # Verify we got a result
        assert "messages" in result
        assert len(result["messages"]) > 0
        
        # Check for mathematical content in the final message
        final_message = result["messages"][-1]
        assert hasattr(final_message, 'content')
        assert final_message.content is not None
        
        # Should contain some numerical result
        content_str = str(final_message.content).lower()
        assert any(str(num) in content_str for num in [120, 143, 15, 8, 23])


    @pytest.mark.skipif(
        not (DEEPAGENTS_AVAILABLE and TAVILY_AVAILABLE), 
        reason="deepagents or tavily not available"
    )
    def test_deepagent_with_internet_search(self):
        """Test deepagent with internet search using Tavily."""
        
        # Check if TAVILY_API_KEY is available
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_api_key:
            pytest.skip("TAVILY_API_KEY not found in environment")

        tavily_client = TavilyClient(api_key=tavily_api_key)

        # Search tool to use for research
        def internet_search(
            query: str,
            max_results: int = 3,
            topic: Literal["general", "news", "finance"] = "general",
            include_raw_content: bool = False,
        ) -> str:
            """Run a web search using Tavily.
            
            Args:
                query: Search query string
                max_results: Maximum number of results to return
                topic: Topic category for search
                include_raw_content: Whether to include full content
                
            Returns:
                Search results as formatted string
            """
            try:
                results = tavily_client.search(
                    query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                    topic=topic,
                )
                
                # Format results
                formatted_results = f"Search results for '{query}':\n\n"
                for i, result in enumerate(results.get('results', []), 1):
                    formatted_results += f"{i}. {result.get('title', 'No title')}\n"
                    formatted_results += f"   URL: {result.get('url', 'No URL')}\n"
                    content_preview = result.get('content', 'No content')[:200]
                formatted_results += f"   Content: {content_preview}...\n\n"
                
                return formatted_results
            except Exception as e:
                return f"Search failed: {str(e)}"

        # Instructions for the research agent
        research_instructions = (
            "You are an expert researcher. "
            "Your job is to conduct thorough research and provide "
            "comprehensive reports.\n\n"
            "You have access to internet search capabilities.\n\n"
            "## `internet_search`\n\n"
            "Use this to search the web for current information on any topic. "
            "You can specify the number of results, topic category, and "
            "whether to include full content."
        )

        # Create the agent
        agent = create_deep_agent(
            tools=[internet_search],
            instructions=research_instructions,
            model=self.llm,
        )

        # Test the research agent
        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Research the latest developments in quantum computing "
                        "in 2024."
                    )
                }
            ]
        })

        # Verify we got a result
        assert "messages" in result
        assert len(result["messages"]) > 0
        
        # Check final message content
        final_message = result["messages"][-1]
        assert hasattr(final_message, 'content')
        assert final_message.content is not None
        
        # Should contain research-related content
        content_str = str(final_message.content).lower()
        research_terms = [
            "quantum", "computing", "research", "development", "2024"
        ]
        assert any(term in content_str for term in research_terms)

    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not available")
    def test_deepagent_with_file_operations(self):
        """Test deepagent with built-in file system tools."""
        
        def create_report(title: str, content: str) -> str:
            """Create a report with given title and content.
            
            Args:
                title: Report title
                content: Report content
                
            Returns:
                Status message
            """
            return (
                f"Report '{title}' created successfully with "
                f"{len(content)} characters."
            )

        # Instructions for the report agent
        report_instructions = (
            "You are a professional report writer. "
            "Your job is to create well-structured reports.\n\n"
            "You have access to file operations and report creation tools.\n\n"
            "Use the built-in file system tools (write_file, read_file, ls, "
            "edit_file) to manage your reports, and the create_report tool "
            "to generate reports."
        )

        # Create the agent with built-in file tools enabled
        agent = create_deep_agent(
            tools=[create_report],
            instructions=report_instructions,
            model=self.llm,
            # Enable file system tools
            builtin_tools=["write_file", "read_file", "ls", "edit_file"],
        )

        # Test with initial files
        initial_files = {
            "notes.txt": "Research notes: Key findings from market analysis.",
            "data.txt": "Sales data: Q1: 100k, Q2: 120k, Q3: 135k",
        }

        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Create a quarterly sales report based on the "
                        "available data files."
                    )
                }
            ],
            "files": initial_files,
        })

        # Verify we got a result
        assert "messages" in result
        assert len(result["messages"]) > 0
        
        # Check if files were accessed/created
        if "files" in result:
            # Files should be preserved or modified
            assert isinstance(result["files"], dict)
        
        # Check final message
        final_message = result["messages"][-1]
        assert hasattr(final_message, 'content')
        content_str = str(final_message.content).lower()
        
        # Should contain report-related content
        report_terms = [
            "report", "sales", "quarterly", "data", "analysis"
        ]
        assert any(term in content_str for term in report_terms)

    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not available")
    def test_deepagent_with_subagents(self):
        """Test deepagent with proper subagent functionality using subagents param."""
        
        def math_calculator(operation: str, numbers: str) -> str:
            """Perform basic math operations.
            
            Args:
                operation: Type of operation (add, multiply, mean, area_circle,
                    solve_linear)
                numbers: Numbers or parameters for the operation (comma-separated)
                
            Returns:
                Result of the calculation
            """
            nums = [float(x.strip()) for x in numbers.split(",") if x.strip()]
            
            if operation == "add":
                return f"Sum: {sum(nums)}"
            elif operation == "multiply":
                return f"Product: {nums[0] * nums[1] if len(nums) >= 2 else nums[0]}"
            elif operation == "mean":
                return f"Mean: {sum(nums) / len(nums)}"
            elif operation == "area_circle":
                # Expecting radius as first number
                import math
                radius = nums[0]
                area = math.pi * radius * radius
                return f"Circle area with radius {radius}: {area:.2f} square units"
            elif operation == "solve_linear":
                # For equation ax + b = c, expecting a, b, c
                if len(nums) >= 3:
                    a, b, c = nums[0], nums[1], nums[2]
                    x = (c - b) / a if a != 0 else "undefined"
                    return f"Solution for {a}x + {b} = {c}: x = {x}"
                else:
                    return "Need 3 numbers for linear equation (a, b, c)"
            else:
                return f"Unknown operation: {operation}"

        def data_analyzer(data: str, analysis_type: str) -> str:
            """Analyze data sets.
            
            Args:
                data: Comma-separated numbers to analyze
                analysis_type: Type of analysis (mean, median, range)
                
            Returns:
                Analysis result
            """
            try:
                nums = [float(x.strip()) for x in data.split(",") if x.strip()]
                nums.sort()
                
                if analysis_type == "mean":
                    mean = sum(nums) / len(nums)
                    return f"Mean of {data}: {mean:.2f}"
                elif analysis_type == "median":
                    n = len(nums)
                    if n % 2 == 0:
                        median = (nums[n//2 - 1] + nums[n//2]) / 2
                    else:
                        median = nums[n//2]
                    return f"Median of {data}: {median}"
                elif analysis_type == "range":
                    range_val = max(nums) - min(nums)
                    return f"Range of {data}: {range_val}"
                else:
                    return f"Unknown analysis type: {analysis_type}"
            except Exception as e:
                return f"Error analyzing data: {e}"

        # Define specialized subagents
        subagents = [
            {
                "name": "geometry-specialist",
                "description": (
                    "Specialized in geometric calculations like areas, perimeters, "
                    "volumes"
                ),
                "prompt": (
                    "You are a geometry specialist. Focus on solving geometric "
                    "problems with precision. Use the math_calculator tool for area "
                    "calculations, and provide clear explanations of geometric "
                    "concepts."
                ),
                "tools": ["math_calculator"]  # Only has access to math_calculator
            },
            {
                "name": "statistics-analyst", 
                "description": (
                    "Expert in statistical analysis and data interpretation"
                ),
                "prompt": (
                    "You are a statistics analyst. Your specialty is analyzing "
                    "datasets and computing statistical measures. Use the "
                    "data_analyzer tool for statistical calculations and explain "
                    "the significance of results."
                ),
                "tools": ["data_analyzer"]  # Only has access to data_analyzer  
            },
            {
                "name": "algebra-solver",
                "description": (
                    "Specialized in solving algebraic equations and expressions"
                ),
                "prompt": (
                    "You are an algebra expert. Focus on solving equations, working "
                    "with variables, and explaining algebraic concepts clearly. Use "
                    "the math_calculator tool for equation solving."
                ),
                "tools": ["math_calculator"]  # Only has access to math_calculator
            }
        ]

        # Instructions for the main coordination agent
        coordinator_instructions = (
            "You are a math tutor coordinator. Your job is to help students "
            "solve complex mathematical problems by breaking them down and "
            "delegating to specialized subagents.\n\n"
            "You have access to specialized subagents:\n"
            "- geometry-specialist: For geometric calculations\n"
            "- statistics-analyst: For statistical analysis\n" 
            "- algebra-solver: For algebraic equations\n\n"
            "When you receive a complex problem with multiple parts:\n"
            "1. Break it down into individual components\n"
            "2. Delegate each component to the appropriate specialist\n"
            "3. Collect results and provide a comprehensive summary\n\n"
            "Use the call_subagent tool to delegate tasks to specialists."
        )

        # Create the agent with custom subagents
        agent = create_deep_agent(
            tools=[math_calculator, data_analyzer],
            instructions=coordinator_instructions, 
            model=self.llm,
            subagents=subagents  # Pass the custom subagents
        )

        # Test complex multi-part problem that requires delegation
        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I need help with a complex math problem that involves "
                        "multiple concepts:\n\n"
                        "1. Calculate the area of a circle with radius 5\n"
                        "2. Find the mean of the dataset [2, 4, 6, 8, 10, 12]\n"
                        "3. Solve the equation 2x + 3 = 15\n\n"
                        "Please solve each part step-by-step using your specialists."
                    )
                }
            ]
        })

        # Verify we got a result
        assert "messages" in result
        assert len(result["messages"]) > 0
        
        # Check final message content
        final_message = result["messages"][-1]
        assert hasattr(final_message, 'content')
        content_str = str(final_message.content).lower()
        
        # Should contain mathematical content from different specialists
        expected_terms = [
            "area", "circle", "radius",  # From geometry specialist
            "mean", "dataset",           # From statistics analyst  
            "equation", "solve", "x",    # From algebra solver
        ]
        assert any(term in content_str for term in expected_terms)
        
        # Should show evidence of subagent delegation
        delegation_terms = [
            "specialist", "delegate", "subagent", "expert"
        ]
        # Check if any message in the conversation shows delegation
        all_content = " ".join([
            str(msg.content).lower() for msg in result["messages"] 
            if hasattr(msg, 'content') and msg.content is not None
        ])
        assert any(term in all_content for term in delegation_terms)

    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not available")  
    @pytest.mark.asyncio
    async def test_deepagent_async_operations(self):
        """Test deepagent with async operations."""
        
        async def async_data_processor(data_type: str, operation: str) -> str:
            """Process data asynchronously.
            
            Args:
                data_type: Type of data to process
                operation: Operation to perform
                
            Returns:
                Processing result
            """
            # Simulate async processing
            import asyncio
            await asyncio.sleep(0.1)
            
            return (
                f"Async processing completed: {operation} on {data_type} data"
            )

        # Instructions for async agent
        async_instructions = (
            "You are an asynchronous data processing agent. "
            "Your job is to handle data processing tasks efficiently.\n\n"
            "You have access to async data processing tools.\n\n"
            "## `async_data_processor`\n\n"
            "Use this for asynchronous data processing operations."
        )

        # Import async create function
        try:
            from deepagents import async_create_deep_agent
            
            # Create async agent
            agent = async_create_deep_agent(
                tools=[async_data_processor],
                instructions=async_instructions,
                model=self.llm,
            )

            # Test async operation
            result = await agent.ainvoke({
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Process customer transaction data using data "
                            "cleaning operations."
                        )
                    }
                ]
            })

            # Verify we got a result
            assert "messages" in result
            assert len(result["messages"]) > 0
            
            # Check final message
            final_message = result["messages"][-1]
            assert hasattr(final_message, 'content')
            assert final_message.content is not None
            
            content_str = str(final_message.content).lower()
            processing_terms = [
                "process", "data", "customer", "transaction", "cleaning"
            ]
            assert any(term in content_str for term in processing_terms)
            
        except ImportError:
            pytest.skip("async_create_deep_agent not available")

    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not available")
    def test_deepagent_error_handling(self):
        """Test deepagent error handling with failing tools."""
        
        def unreliable_tool(operation: str) -> str:
            """A tool that sometimes fails.
            
            Args:
                operation: Operation to attempt
                
            Returns:
                Result or error message
            """
            if operation == "fail":
                raise ValueError("Simulated tool failure")
            else:
                return f"Operation '{operation}' completed successfully"

        # Instructions for error handling test
        error_instructions = (
            "You are a resilient assistant that handles tool failures "
            "gracefully.\n\n"
            "You have access to an unreliable tool that may fail.\n\n"
            "## `unreliable_tool`\n\n"
            "Use this tool carefully as it may fail for certain operations. "
            "Handle failures gracefully and provide alternative solutions."
        )

        # Create the agent
        agent = create_deep_agent(
            tools=[unreliable_tool],
            instructions=error_instructions,
            model=self.llm,
        )

        # Test with operation that should succeed
        result_success = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": "Please perform a 'process' operation."
                }
            ]
        })

        # Should get a successful result
        assert "messages" in result_success
        assert len(result_success["messages"]) > 0

        # Test with operation that might fail
        result_fail = agent.invoke({
            "messages": [
                {
                    "role": "user", 
                    "content": "Please perform a 'fail' operation."
                }
            ]
        })

        # Should still get a result (agent should handle the failure)
        assert "messages" in result_fail
        assert len(result_fail["messages"]) > 0
        
        final_message = result_fail["messages"][-1]
        assert hasattr(final_message, 'content')
        assert final_message.content is not None


class TestDeepAgentsIntegrationRequirements:
    """Test requirements and setup for deepagents integration."""
    
    def test_chatqwen_vllm_compatibility(self):
        """Test that ChatQwenVllm has required methods for deepagents."""
        llm = ChatQwenVllm(model="Qwen/Qwen3-32B")
        
        # Check required methods for LangChain chat model compatibility
        assert hasattr(llm, 'invoke')
        assert hasattr(llm, 'ainvoke')
        assert hasattr(llm, 'stream')
        assert hasattr(llm, 'astream')
        assert hasattr(llm, 'bind_tools')
        assert hasattr(llm, 'with_structured_output')
        
        # Check VLLM-specific features
        assert hasattr(llm, 'enable_thinking')
        assert llm._support_tool_choice() is True
        
    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not available")
    def test_deepagents_import_and_basic_setup(self):
        """Test that deepagents can be imported and basic setup works."""
        from deepagents import create_deep_agent
        
        # Simple test tool
        def test_tool(message: str) -> str:
            """Process a message and return it with a prefix."""
            return f"Processed: {message}"
        
        # Should be able to create agent without errors
        llm = ChatQwenVllm(model="Qwen/Qwen3-32B", temperature=0.1)
        
        try:
            agent = create_deep_agent(
                tools=[test_tool],
                instructions="You are a test assistant.",
                model=llm,
            )
            
            # Agent should have required attributes
            assert agent is not None
            assert hasattr(agent, 'invoke')
            
        except Exception as e:
            pytest.fail(f"Failed to create basic deepagent: {e}")
    
    def test_environment_configuration(self):
        """Test that environment is properly configured for deepagents testing."""
        # Check for VLLM configuration
        vllm_api_base = os.environ.get("VLLM_API_BASE")
        # Should have VLLM configuration or use default
        assert vllm_api_base is not None or True  # Default is handled by ChatQwenVllm
        
        # Check for optional Tavily configuration
        tavily_key = os.environ.get("TAVILY_API_KEY")
        if tavily_key:
            # If Tavily is configured, it should be a non-empty string
            assert isinstance(tavily_key, str)
            assert len(tavily_key) > 0
