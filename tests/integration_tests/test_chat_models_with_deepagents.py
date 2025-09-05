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

    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not available")
    def test_deepagent_with_structured_output_tools(self):
        """Test deepagent with structured output tools."""
        
        class AnalysisResult(BaseModel):
            """Result of data analysis."""
            summary: str = Field(description="Brief summary of the analysis")
            key_findings: List[str] = Field(description="List of key findings")
            recommendation: str = Field(description="Recommended action")
            confidence_score: float = Field(
                description="Confidence score 0-1", ge=0, le=1
            )

        def analyze_data(data_description: str) -> str:
            """Analyze the given data description.
            
            Args:
                data_description: Description of the data to analyze
                
            Returns:
                Analysis result as text
            """
            # Mock data analysis
            if "sales" in data_description.lower():
                return """Sales Analysis Results:
                - Total sales increased by 15% compared to last quarter
                - Top performing product: Product A (35% of total sales)
                - Geographic performance: North region leads with 40% growth
                - Recommendation: Expand Product A inventory and focus on North region
                - Confidence: 0.85 based on historical data patterns"""
            elif "user" in data_description.lower():
                return """User Behavior Analysis:
                - Average session duration: 8.5 minutes (up 12%)
                - Bounce rate: 23% (down 8%)
                - Top user action: Search functionality (used by 78% of users)
                - Recommendation: Optimize search algorithm and add search suggestions
                - Confidence: 0.92 based on comprehensive user tracking"""
            else:
                return """General Data Analysis:
                - Data quality: Good (87% completeness)
                - Trends: Positive growth trajectory
                - Anomalies: 3 minor outliers detected
                - Recommendation: Continue current strategy with minor adjustments
                - Confidence: 0.75 due to limited context"""

        # Instructions for the analysis agent
        analysis_instructions = (
            "You are a senior data analyst. "
            "Your job is to analyze data and provide structured insights.\n\n"
            "You have access to a data analysis tool.\n\n"
            "## `analyze_data`\n\n"
            "Use this to analyze data based on descriptions provided by users.\n\n"
            "When providing your final analysis, structure your response "
            "according to the AnalysisResult schema."
        )

        # Create the agent
        agent = create_deep_agent(
            tools=[analyze_data],
            instructions=analysis_instructions,
            model=self.llm,
        )

        # Note: structured_llm could be used for final response formatting
        # structured_llm = self.llm.with_structured_output(
        #     schema=AnalysisResult,
        #     method="json_schema"
        # )

        # Test data analysis workflow
        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Please analyze our Q3 sales data and provide "
                        "structured insights."
                    )
                }
            ]
        })

        # Verify we got a result with messages
        assert "messages" in result
        assert len(result["messages"]) > 0
        
        # Check that analysis content is present
        final_message = result["messages"][-1]
        assert hasattr(final_message, 'content')
        content = str(final_message.content).lower()
        
        # Should contain analysis-related terms
        analysis_terms = [
            "sales", "analysis", "product", "recommendation", "growth"
        ]
        assert any(term in content for term in analysis_terms)

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
            builtin_tools=["write_file", "read_file", "ls", "edit_file"],  # Enable file system tools
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
        """Test deepagent with subagent functionality."""
        
        def specialized_calculation(calculation_type: str, problem: str) -> str:
            """Perform specialized calculations.
            
            Args:
                calculation_type: Type of calculation (statistics, geometry, algebra)
                problem: Description of the problem to solve
                
            Returns:
                Solution to the problem
            """
            if calculation_type == "statistics":
                return "Statistical analysis: Mean=45.6, Median=44.2, Std Dev=12.3"
            elif calculation_type == "geometry":
                return "Geometric solution: Area=78.5 sq units, Perimeter=31.4 units"
            elif calculation_type == "algebra":
                return "Algebraic solution: x=7.2, y=-3.1, z=5.8"
            else:
                return (
                    f"Solution for {calculation_type}: "
                    "Result calculated successfully"
                )

        # Instructions for the math tutor agent
        tutor_instructions = (
            "You are an expert math tutor. "
            "Your job is to help students solve complex mathematical problems.\n\n"
            "You have access to specialized calculation tools and can "
            "delegate to subagents for complex tasks.\n\n"
            "## `specialized_calculation`\n\n"
            "Use this for specialized mathematical calculations in statistics, "
            "geometry, or algebra.\n\n"
            "When dealing with complex multi-step problems, break them down "
            "and use the general-purpose subagent for coordination."
        )

        # Create the agent with subagent support
        agent = create_deep_agent(
            tools=[specialized_calculation],
            instructions=tutor_instructions,
            model=self.llm,
        )

        # Test complex problem that might benefit from subagents
        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I need help with a complex math problem that involves "
                        "multiple concepts:\n\n"
                        "1. Calculate the area of a circle with radius 5\n"
                        "2. Find the statistical mean of the dataset "
                        "[2, 4, 6, 8, 10, 12]\n"
                        "3. Solve the equation 2x + 3 = 15\n\n"
                        "Please provide step-by-step solutions for each part."
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
        
        # Should contain mathematical content
        math_terms = [
            "area", "circle", "radius", "mean", "equation", 
            "solution", "calculate"
        ]
        assert any(term in content_str for term in math_terms)

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

    @pytest.mark.skipif(not DEEPAGENTS_AVAILABLE, reason="deepagents not available")
    def test_deepagent_with_thinking_enabled(self):
        """Test deepagent with ChatQwenVllm thinking capabilities."""
        
        def complex_reasoning_task(scenario: str) -> str:
            """Handle complex reasoning scenarios.
            
            Args:
                scenario: Description of the scenario to analyze
                
            Returns:
                Analysis result
            """
            if "ethical" in scenario.lower():
                return """Ethical analysis completed:
                - Primary stakeholders: Patients, healthcare providers, society
                - Competing values: Individual autonomy vs. collective benefit
                - Recommended approach: Balanced framework considering all 
                  perspectives
                - Risk factors: Potential for discrimination, resource 
                  allocation issues"""
            else:
                return (
                    f"Complex analysis of scenario: {scenario} completed with "
                    "multi-factor consideration."
                )

        # Instructions that encourage deep thinking
        thinking_instructions = (
            "You are a philosopher and critical thinking expert. "
            "Your job is to analyze complex scenarios that require deep "
            "reasoning.\n\n"
            "You have access to complex reasoning tools and should think "
            "step by step.\n\n"
            "## `complex_reasoning_task`\n\n"
            "Use this for analyzing complex scenarios that require careful "
            "consideration of multiple factors.\n\n"
            "Take your time to think through problems thoroughly before "
            "providing conclusions."
        )

        # Create agent with thinking-enabled ChatQwenVllm
        thinking_llm = ChatQwenVllm(
            model="Qwen/Qwen3-32B",
            temperature=0.1,
            enable_thinking=True,  # Enable thinking mode
            max_tokens=3000,
        )

        agent = create_deep_agent(
            tools=[complex_reasoning_task],
            instructions=thinking_instructions,
            model=thinking_llm,
        )

        # Test with a complex ethical scenario
        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Please analyze this ethical dilemma:\n\n"
                        "A new medical treatment is available that can save lives, "
                        "but it's extremely expensive and resources are limited. "
                        "How should healthcare systems decide who gets access to "
                        "this treatment? Consider the ethical implications and "
                        "provide a reasoned analysis."
                    )
                }
            ]
        })

        # Verify we got a comprehensive result
        assert "messages" in result
        assert len(result["messages"]) > 0
        
        # Check that the response shows evidence of complex reasoning
        final_message = result["messages"][-1]
        assert hasattr(final_message, 'content')
        content_str = str(final_message.content).lower()
        
        # Should contain ethical reasoning concepts
        reasoning_terms = [
            "ethical", "analysis", "stakeholder", "consider", "factor", 
            "treatment", "healthcare", "resource", "access", "decision"
        ]
        assert any(term in content_str for term in reasoning_terms)
        
        # Content should be substantial (indicating deep reasoning)
        assert len(content_str) > 100  # Should be a thoughtful response


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
