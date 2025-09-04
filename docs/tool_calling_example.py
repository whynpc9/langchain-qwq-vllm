#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: ChatQwenVllm Tool Calling with Weather Query

This example demonstrates how to use ChatQwenVllm with tool calling functionality,
including both text output and structured output scenarios.
"""

from langchain_core.tools import tool
from langchain_qwq import ChatQwenVllm
from pydantic import BaseModel, Field
from typing import List


# Define the weather tool
@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a specific location.
    
    Args:
        location: The city or location to get weather for
        unit: Temperature unit, either 'celsius' or 'fahrenheit'
    
    Returns:
        A string describing the current weather conditions
    """
    # Mock weather data for demonstration
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


# Define schemas for structured output
class WeatherInfo(BaseModel):
    """Individual weather information."""
    location: str = Field(description="The queried location")
    temperature: int = Field(description="Temperature in degrees")
    condition: str = Field(description="Weather condition")
    humidity: int = Field(description="Humidity percentage", ge=0, le=100)
    unit: str = Field(
        description="Temperature unit: must be exactly 'celsius' or 'fahrenheit'",
        default="celsius"
    )


class WeatherSummary(BaseModel):
    """Summary of multiple weather queries."""
    cities: List[str] = Field(description="List of cities queried")
    weather_reports: List[WeatherInfo] = Field(description="Individual weather reports")
    overall_summary: str = Field(description="Overall weather summary")


def example_text_output():
    """Example 1: Basic tool calling with text output."""
    print("=" * 60)
    print("Example 1: Tool Calling with Text Output")
    print("=" * 60)
    
    # Initialize ChatQwenVllm
    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        api_base="http://localhost:8000/v1",
        temperature=0.1,
        max_tokens=500,
    )
    
    # Bind the weather tool
    llm_with_tools = llm.bind_tools([get_weather])
    
    # Query weather for a single city
    print("Query: What's the weather like in Beijing today?")
    response = llm_with_tools.invoke("What's the weather like in Beijing today?")
    
    print(f"\nResponse type: {type(response)}")
    print(f"Content: {response.content}")
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"Tool calls: {len(response.tool_calls)}")
        for i, tool_call in enumerate(response.tool_calls):
            print(f"  Tool call {i+1}: {tool_call}")
    
    print("\n")


def example_structured_output():
    """Example 2: Tool calling with structured output."""
    print("=" * 60)
    print("Example 2: Tool Calling with Structured Output")
    print("=" * 60)
    
    # Initialize ChatQwenVllm
    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        api_base="http://localhost:8000/v1",
        temperature=0.1,
        max_tokens=1000,
    )
    
    # First bind tools, then add structured output
    llm_with_tools = llm.bind_tools([get_weather])
    structured_llm = llm_with_tools.with_structured_output(
        schema=WeatherInfo,
        method="json_schema"
    )
    
    # Query with structured output requirement
    print("Query: Get weather information for Tokyo in structured format")
    
    messages = [
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
    ]
    
    response = structured_llm.invoke(messages)
    
    print(f"\nResponse type: {type(response)}")
    if isinstance(response, WeatherInfo):
        print(f"Location: {response.location}")
        print(f"Temperature: {response.temperature}°{response.unit[0].upper()}")
        print(f"Condition: {response.condition}")
        print(f"Humidity: {response.humidity}%")
        print(f"Unit: {response.unit}")
    else:
        print(f"Response: {response}")
    
    print("\n")


def example_multiple_cities():
    """Example 3: Multiple cities with structured summary."""
    print("=" * 60)
    print("Example 3: Multiple Cities with Structured Summary")
    print("=" * 60)
    
    # Initialize ChatQwenVllm
    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        api_base="http://localhost:8000/v1",
        temperature=0.1,
        max_tokens=1500,
    )
    
    # Bind tools and add structured output
    llm_with_tools = llm.bind_tools([get_weather])
    structured_llm = llm_with_tools.with_structured_output(
        schema=WeatherSummary,
        method="json_schema"
    )
    
    print("Query: Get weather for Beijing, Shanghai, and Tokyo with summary")
    
    messages = [
        {
            "role": "system",
            "content": "You are a weather analyst. Use the weather tool to gather information for multiple cities, then provide a comprehensive summary in the required format."
        },
        {
            "role": "user",
            "content": "Please get weather information for Beijing, Shanghai, and Tokyo. Provide individual reports and an overall summary."
        }
    ]
    
    response = structured_llm.invoke(messages)
    
    print(f"\nResponse type: {type(response)}")
    if isinstance(response, WeatherSummary):
        print(f"Cities queried: {', '.join(response.cities)}")
        print(f"Number of reports: {len(response.weather_reports)}")
        
        print("\nIndividual Weather Reports:")
        for i, report in enumerate(response.weather_reports, 1):
            print(f"  {i}. {report.location}: {report.condition}, {report.temperature}°{report.unit[0].upper()}, {report.humidity}% humidity")
        
        print(f"\nOverall Summary: {response.overall_summary}")
    else:
        print(f"Response: {response}")
    
    print("\n")


def example_with_raw_output():
    """Example 4: Structured output with raw response included."""
    print("=" * 60)
    print("Example 4: Structured Output with Raw Response")
    print("=" * 60)
    
    # Initialize ChatQwenVllm
    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        api_base="http://localhost:8000/v1",
        temperature=0.1,
        max_tokens=800,
    )
    
    # Bind tools and add structured output with include_raw=True
    llm_with_tools = llm.bind_tools([get_weather])
    structured_llm = llm_with_tools.with_structured_output(
        schema=WeatherInfo,
        method="json_schema",
        include_raw=True
    )
    
    print("Query: Weather in London with raw output included")
    response = structured_llm.invoke("What's the weather like in London?")
    
    print(f"\nResponse type: {type(response)}")
    if isinstance(response, dict):
        print("Raw response available:", "raw" in response)
        print("Parsed response available:", "parsed" in response)
        print("Parsing error:", response.get("parsing_error"))
        
        if response["parsing_error"] is None and response["parsed"]:
            parsed = response["parsed"]
            print(f"\nParsed Weather Info:")
            print(f"  Location: {parsed.location}")
            print(f"  Temperature: {parsed.temperature}°{parsed.unit[0].upper()}")
            print(f"  Condition: {parsed.condition}")
            print(f"  Humidity: {parsed.humidity}%")
        
        if response["raw"]:
            print(f"\nRaw response content length: {len(str(response['raw'].content))}")
    
    print("\n")


async def example_async_tool_calling():
    """Example 5: Async tool calling."""
    print("=" * 60)
    print("Example 5: Async Tool Calling")
    print("=" * 60)
    
    # Initialize ChatQwenVllm
    llm = ChatQwenVllm(
        model="Qwen/Qwen3-32B",
        api_base="http://localhost:8000/v1",
        temperature=0.1,
        max_tokens=500,
    )
    
    # Bind tools
    llm_with_tools = llm.bind_tools([get_weather])
    
    print("Async Query: Weather in New York")
    response = await llm_with_tools.ainvoke("What's the weather in New York?")
    
    print(f"\nAsync Response type: {type(response)}")
    print(f"Content: {response.content}")
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"Tool calls: {len(response.tool_calls)}")
        for i, tool_call in enumerate(response.tool_calls):
            print(f"  Tool call {i+1}: {tool_call}")
    
    print("\n")


def main():
    """Run all examples."""
    print("ChatQwenVllm Tool Calling Examples")
    print("Note: These examples require a running vLLM server")
    print("Start server with: vllm serve Qwen/Qwen3-32B --port 8000")
    print()
    
    try:
        # Run synchronous examples
        example_text_output()
        example_structured_output()
        example_multiple_cities()
        example_with_raw_output()
        
        # Run async example
        import asyncio
        asyncio.run(example_async_tool_calling())
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure your vLLM server is running at http://localhost:8000")


if __name__ == "__main__":
    main()
