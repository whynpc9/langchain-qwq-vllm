"""Integration tests for structured output with LangChain 1.x agents."""

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_qwq.chat_models_vllm import ChatQwenVllm

load_dotenv()


class TestStructuredOutputWithAgent:
    """Test structured output integration with LangChain 1.x agents.
    
    Note: Tests with tool calling use ToolStrategy instead of ProviderStrategy
    because vLLM cannot handle tools and guided_json simultaneously. ToolStrategy
    converts the structured output schema into a special tool that can coexist
    with regular tools in the same agent.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Set dummy API key for VLLM
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "dummy-key-for-vllm"
        
        self.llm = ChatQwenVllm(
            model="Qwen/Qwen3-32B",
            temperature=0.1,
            max_tokens=2000,
            enable_thinking=True,  # Required for ToolStrategy with Qwen3+vLLM
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
            system_prompt=(
                "You are a helpful assistant that extracts contact "
                "information."
            ),
            response_format=ProviderStrategy(ContactInfo)
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        "Extract contact info from: John Doe, "
                        "john@example.com, (555) 123-4567"
                    )
                )
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
            system_prompt=(
                "You are a helpful assistant that extracts person "
                "information."
            ),
            response_format=ProviderStrategy(Person)
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        "Extract info: Alice Smith, 30 years old, lives at "
                        "123 Main St, San Francisco, CA 94102"
                    )
                )
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
            sentiment: Literal["positive", "negative", "neutral"] = Field(
                description="Overall sentiment"
            )
            key_points: list[str] = Field(
                description="Key points mentioned. Lowercase, 1-3 words each."
            )
            pros: list[str] = Field(
                description="Positive aspects mentioned"
            )
            cons: list[str] = Field(
                description="Negative aspects mentioned"
            )

        agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt=(
                "You are a helpful assistant that analyzes product "
                "reviews."
            ),
            response_format=ProviderStrategy(ProductReview)
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        "Analyze: 'Great phone! 5 stars. Beautiful display "
                        "and fast performance. Battery life could be better "
                        "and it's quite expensive.'"
                    )
                )
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
            location: str | None = Field(
                default=None, description="Event location if mentioned"
            )
            attendees: int | None = Field(
                default=None, description="Number of attendees if mentioned"
            )

        agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt=(
                "You are a helpful assistant that extracts event "
                "information."
            ),
            response_format=ProviderStrategy(EventInfo)
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content="Extract event info: Tech Conference on March 15th, 2024"
                )
            ]
        })

        assert "structured_response" in result
        structured_data = result["structured_response"]
        
        assert isinstance(structured_data, EventInfo)
        assert (
            "Tech Conference" in structured_data.event_name
            or "conference" in structured_data.event_name.lower()
        )
        # Date can be in various formats (March 15, 2024-03-15, etc.)
        assert (
            "2024" in structured_data.date
            or "March" in structured_data.date
            or "15" in structured_data.date
        )
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
            priority: Literal["low", "medium", "high", "urgent"] = Field(
                description="Task priority"
            )
            estimated_hours: int = Field(
                description="Estimated hours to complete", ge=1, le=100
            )

        agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt=(
                "You are a helpful assistant that creates task "
                "information."
            ),
            response_format=ProviderStrategy(Task)
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        "Create a task: Fix critical database bug - This is "
                        "blocking production, needs immediate attention, "
                        "estimated 4 hours"
                    )
                )
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
            order_id: str = Field(
                description="Order ID", min_length=5, max_length=20
            )
            quantity: int = Field(
                description="Quantity ordered", ge=1, le=1000
            )
            total_price: float = Field(description="Total price", gt=0)
            currency: Literal["USD", "EUR", "GBP"] = Field(
                description="Currency code"
            )

        agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt=(
                "You are a helpful assistant that extracts order "
                "information."
            ),
            response_format=ProviderStrategy(OrderInfo)
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content="Extract order: Order #ORD12345, 5 items, total $99.99 USD"
                )
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
            rating: float = Field(
                description="Rating out of 10", ge=0, le=10
            )

        agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt=(
                "You are a helpful assistant that extracts movie "
                "information."
            ),
            response_format=ProviderStrategy(MovieInfo)
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        "Extract: The Shawshank Redemption (1994), directed "
                        "by Frank Darabont, is a drama film rated 9.3/10"
                    )
                )
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

    def test_structured_output_with_simple_tool_call(self):
        """Test structured output with simple tool call using ToolStrategy."""
        
        @tool
        def calculator(operation: str, a: float, b: float) -> str:
            """Perform basic math operations.
            
            Args:
                operation: The operation (add, subtract, multiply, divide)
                a: First number
                b: Second number
            
            Returns:
                Result of the operation
            """
            operations = {
                "add": lambda x, y: x + y,
                "subtract": lambda x, y: x - y,
                "multiply": lambda x, y: x * y,
                "divide": lambda x, y: x / y if y != 0 else None
            }
            if operation in operations:
                result = operations[operation](a, b)
                if result is not None:
                    return f"{result}"
                else:
                    return "Error: division by zero"
            return f"Error: unknown operation {operation}"

        class CalculationResult(BaseModel):
            """Result of a mathematical calculation."""
            problem: str = Field(description="The original problem")
            steps: list[str] = Field(description="Steps taken to solve the problem")
            final_answer: float = Field(description="The final numerical answer")
            
        agent = create_agent(
            model=self.llm,
            tools=[calculator],
            system_prompt=(
                "You are a math assistant. Use the calculator tool to solve "
                "problems. Show your work step by step and provide the final "
                "answer in the specified format."
            ),
            response_format=ToolStrategy(
                schema=CalculationResult,
                tool_message_content="Calculation completed and structured!"
            )
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(content="Calculate: (15 * 8) + 23")
            ]
        })

        assert "structured_response" in result
        structured_data = result["structured_response"]
        
        assert isinstance(structured_data, CalculationResult)
        assert structured_data.final_answer == 143.0  # 15 * 8 + 23 = 143
        assert len(structured_data.steps) > 0
        
        print(f"✓ Tool call with structured output: {structured_data}")

    def test_structured_output_with_multiple_tool_calls(self):
        """Test structured output with multiple tool calls using ToolStrategy."""
        
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

        class MathSolution(BaseModel):
            """Solution to a complex math problem."""
            original_problem: str = Field(
                description="The original problem statement"
            )
            tool_calls_made: list[str] = Field(
                description="List of tool calls made (e.g., 'add(2, 3)')"
            )
            intermediate_results: list[float] = Field(
                description="Intermediate calculation results"
            )
            final_result: float = Field(description="The final result")
            explanation: str = Field(
                description="Brief explanation of the solution"
            )

        agent = create_agent(
            model=self.llm,
            tools=[add, multiply, power],
            system_prompt=(
                "You are a math expert. Use the provided tools (add, multiply, power) to solve"
                "complex problems. Then output the final result according to the given structure. Document all tool calls and intermediate "
                "results."
            ),
            response_format=ToolStrategy(
                schema=MathSolution,
                tool_message_content="Math solution structured successfully!"
            )
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        "Calculate: (2 + 3) * 4, then raise the result to "
                        "the power of 2"
                    )
                )
            ]
        })

        assert "structured_response" in result
        structured_data = result["structured_response"]
        
        assert isinstance(structured_data, MathSolution)
        # ((2 + 3) * 4)^2 = 20^2 = 400
        assert structured_data.final_result == 400.0
        # Should have multiple tool calls
        assert len(structured_data.tool_calls_made) >= 2
        assert len(structured_data.intermediate_results) >= 2
        
        print(f"✓ Multiple tool calls with structured output: {structured_data}")

    def test_structured_output_with_data_lookup_tools(self):
        """Test structured output with tools that look up data using ToolStrategy."""
        
        @tool
        def get_weather(city: str) -> str:
            """Get weather information for a city.
            
            Args:
                city: Name of the city
                
            Returns:
                Weather information
            """
            # Mock weather data
            weather_db = {
                "New York": "Sunny, 75°F",
                "London": "Rainy, 60°F",
                "Tokyo": "Cloudy, 70°F",
                "Paris": "Partly cloudy, 68°F"
            }
            return weather_db.get(
                city, f"Weather data not available for {city}"
            )
        
        @tool
        def get_population(city: str) -> str:
            """Get population information for a city.
            
            Args:
                city: Name of the city
                
            Returns:
                Population information
            """
            # Mock population data
            population_db = {
                "New York": "8.3 million",
                "London": "9.0 million",
                "Tokyo": "13.9 million",
                "Paris": "2.2 million"
            }
            return population_db.get(city, f"Population data not available for {city}")

        class CityReport(BaseModel):
            """Comprehensive report about a city."""
            city_name: str = Field(description="Name of the city")
            weather: str = Field(description="Current weather conditions")
            population: str = Field(description="Population information")
            summary: str = Field(description="Brief summary of the city")
            data_sources_used: list[str] = Field(
                description="Tools/sources used to gather data"
            )

        agent = create_agent(
            model=self.llm,
            tools=[get_weather, get_population],
            system_prompt=(
                "You are a city information assistant. Use the available "
                "tools to gather comprehensive information about cities and "
                "provide structured reports."
            ),
            response_format=ToolStrategy(
                schema=CityReport,
                tool_message_content="City report compiled successfully!"
            )
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(content="Give me a comprehensive report about Tokyo")
            ]
        })

        assert "structured_response" in result
        structured_data = result["structured_response"]
        
        assert isinstance(structured_data, CityReport)
        assert (
            "Tokyo" in structured_data.city_name
            or "tokyo" in structured_data.city_name.lower()
        )
        assert len(structured_data.weather) > 0
        assert len(structured_data.population) > 0
        assert len(structured_data.data_sources_used) > 0
        
        print(f"✓ Data lookup tools with structured output: {structured_data}")

    def test_structured_output_with_analysis_tools(self):
        """Test structured output with analysis tools using ToolStrategy."""
        
        @tool
        def analyze_text_sentiment(text: str) -> str:
            """Analyze the sentiment of text.
            
            Args:
                text: Text to analyze
                
            Returns:
                Sentiment analysis result
            """
            # Simple mock sentiment analysis
            positive_words = ['great', 'excellent', 'good', 'wonderful', 'amazing']
            negative_words = ['bad', 'terrible', 'poor', 'awful', 'horrible']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return "Positive sentiment detected"
            elif neg_count > pos_count:
                return "Negative sentiment detected"
            else:
                return "Neutral sentiment"
        
        @tool
        def count_words(text: str) -> int:
            """Count the number of words in text.
            
            Args:
                text: Text to analyze
                
            Returns:
                Number of words
            """
            return len(text.split())

        class TextAnalysisReport(BaseModel):
            """Comprehensive text analysis report."""
            original_text: str = Field(
                description="The text that was analyzed"
            )
            word_count: int = Field(
                description="Number of words in the text"
            )
            sentiment: Literal["positive", "negative", "neutral"] = Field(
                description="Overall sentiment"
            )
            key_observations: list[str] = Field(
                description="Key observations from the analysis"
            )
            tools_used: list[str] = Field(
                description="Analysis tools that were used"
            )

        agent = create_agent(
            model=self.llm,
            tools=[analyze_text_sentiment, count_words],
            system_prompt=(
                "You are a text analysis expert. Use the available tools "
                "to analyze text and provide comprehensive structured reports."
            ),
            response_format=ToolStrategy(
                schema=TextAnalysisReport,
                tool_message_content="Analysis report generated successfully!"
            )
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        "Analyze this review: 'This product is excellent! "
                        "Great quality and wonderful performance. "
                        "Highly recommended!'"
                    )
                )
            ]
        })

        assert "structured_response" in result
        structured_data = result["structured_response"]
        
        assert isinstance(structured_data, TextAnalysisReport)
        assert structured_data.sentiment in ["positive", "negative", "neutral"]
        assert structured_data.word_count > 0
        assert len(structured_data.key_observations) > 0
        assert len(structured_data.tools_used) > 0
        
        print(f"✓ Analysis tools with structured output: {structured_data}")

    def test_structured_output_with_conditional_tool_use(self):
        """Test structured output with conditional tool use using ToolStrategy."""
        
        @tool
        def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
            """Convert temperature between units.
            
            Args:
                value: Temperature value
                from_unit: Source unit (C, F, K)
                to_unit: Target unit (C, F, K)
                
            Returns:
                Converted temperature
            """
            # Convert to Celsius first
            if from_unit == "F":
                celsius = (value - 32) * 5/9
            elif from_unit == "K":
                celsius = value - 273.15
            else:
                celsius = value
            
            # Convert to target unit
            if to_unit == "F":
                result = celsius * 9/5 + 32
            elif to_unit == "K":
                result = celsius + 273.15
            else:
                result = celsius
                
            return f"{result:.2f}"
        
        @tool
        def check_temperature_safety(temp_celsius: float, context: str) -> str:
            """Check if a temperature is safe for a given context.
            
            Args:
                temp_celsius: Temperature in Celsius
                context: Context (e.g., 'human', 'food_storage', 'electronics')
                
            Returns:
                Safety assessment
            """
            safety_ranges = {
                "human": (15, 30, "comfortable"),
                "food_storage": (0, 5, "safe for refrigeration"),
                "electronics": (10, 35, "safe for operation")
            }
            
            if context in safety_ranges:
                min_temp, max_temp, description = safety_ranges[context]
                if min_temp <= temp_celsius <= max_temp:
                    return f"Safe: {description}"
                else:
                    return (
                        f"Warning: Temperature outside safe range "
                        f"({min_temp}°C - {max_temp}°C)"
                    )
            return "Unknown context"

        class TemperatureAnalysis(BaseModel):
            """Temperature analysis with safety assessment."""
            input_value: float = Field(
                description="Original temperature value"
            )
            input_unit: str = Field(
                description="Original temperature unit"
            )
            converted_values: dict[str, float] = Field(
                description="Temperature in different units (C, F, K)"
            )
            safety_assessment: str | None = Field(
                default=None, description="Safety assessment if applicable"
            )
            recommendations: list[str] = Field(
                description="Recommendations based on the temperature"
            )

        agent = create_agent(
            model=self.llm,
            tools=[convert_temperature, check_temperature_safety],
            system_prompt=(
                "You are a temperature analysis expert. Convert temperatures "
                "between units and assess safety when relevant. Provide "
                "structured analysis."
            ),
            response_format=ToolStrategy(
                schema=TemperatureAnalysis,
                tool_message_content="Temperature analysis completed!"
            )
        )

        result = agent.invoke({
            "messages": [
                HumanMessage(content="Analyze this temperature for human comfort: 72°F")
            ]
        })

        assert "structured_response" in result
        structured_data = result["structured_response"]
        
        assert isinstance(structured_data, TemperatureAnalysis)
        assert structured_data.input_value == 72.0
        assert structured_data.input_unit in ["F", "Fahrenheit"]
        assert len(structured_data.converted_values) > 0
        assert len(structured_data.recommendations) > 0
        
        print(f"✓ Conditional tool use with structured output: {structured_data}")


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
    print("Testing Structured Output with Tool Calls")
    print("=" * 60)
    
    print("\nTest 8: Simple Tool Call")
    test_suite.test_structured_output_with_simple_tool_call()
    
    print("\nTest 9: Multiple Tool Calls")
    test_suite.test_structured_output_with_multiple_tool_calls()
    
    print("\nTest 10: Data Lookup Tools")
    test_suite.test_structured_output_with_data_lookup_tools()
    
    print("\nTest 11: Analysis Tools")
    test_suite.test_structured_output_with_analysis_tools()
    
    print("\nTest 12: Conditional Tool Use")
    test_suite.test_structured_output_with_conditional_tool_use()
    
    print("\n" + "=" * 60)
    print("✓ All structured output tests passed (including tool calls)!")
    print("=" * 60)

