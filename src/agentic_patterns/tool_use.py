"""
Tool Use Pattern Implementation.

Based on the Agentic Design Patterns book Chapter 5:
Enable agents to interact with external APIs, databases, or services.

The Tool Use pattern allows an LLM to:
1. Tool Definition - Understand available tools and their parameters
2. LLM Decision - Decide when to use a tool based on the request
3. Function Call Generation - Generate structured calls to tools
4. Tool Execution - Execute the tool via the framework
5. Observation/Result - Return results to the agent

Key implementation: Weather, Calculator, and Search tools with an agent
that orchestrates tool calls based on user requests.
"""

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai import RunContext

from agentic_patterns._models import get_model


# --8<-- [start:models]
class WeatherResult(BaseModel):
    """Result from weather tool."""

    location: str = Field(description="The queried location")
    temperature: float = Field(description="Temperature in Celsius")
    conditions: str = Field(description="Weather conditions description")
    humidity: int = Field(ge=0, le=100, description="Humidity percentage")


class CalculationResult(BaseModel):
    """Result from calculation tool."""

    expression: str = Field(description="The evaluated expression")
    result: float = Field(description="The calculation result")
    formatted: str = Field(description="Human-readable result")


class SearchResult(BaseModel):
    """Result from search tool."""

    query: str = Field(description="The search query")
    results: list[str] = Field(
        default_factory=list, description="Search result summaries"
    )
    source_count: int = Field(description="Number of sources found")


class ToolResponse(BaseModel):
    """Final response from the tool-using agent."""

    answer: str = Field(description="The final answer to the user's query")
    tools_used: list[str] = Field(
        default_factory=list, description="List of tools that were invoked"
    )
    reasoning: str = Field(description="Brief explanation of the approach")


@dataclass
class ToolDependencies:
    """Dependencies for tool execution."""

    # Simulated external data sources
    weather_data: dict[str, dict] | None = None
    search_data: dict[str, list[str]] | None = None


# --8<-- [end:models]


# --8<-- [start:agent]
# Initialize the model
model = get_model()

# Tool-using agent
tool_agent = Agent(
    model,
    system_prompt=(
        "You are a helpful assistant with access to tools. "
        "Use the available tools to answer user questions accurately. "
        "Always explain which tools you used and why."
    ),
    deps_type=ToolDependencies,
    output_type=ToolResponse,
)
# --8<-- [end:agent]


# --8<-- [start:tools]
# Default simulated weather data
DEFAULT_WEATHER_DATA = {
    "london": {
        "temperature": 15.0,
        "conditions": "Cloudy with a chance of rain",
        "humidity": 78,
    },
    "paris": {
        "temperature": 18.5,
        "conditions": "Partly sunny",
        "humidity": 65,
    },
    "tokyo": {
        "temperature": 22.0,
        "conditions": "Clear skies",
        "humidity": 55,
    },
    "new york": {
        "temperature": 12.0,
        "conditions": "Overcast",
        "humidity": 70,
    },
}

# Default simulated search data
DEFAULT_SEARCH_DATA = {
    "python programming": [
        "Python is a high-level programming language",
        "Python supports multiple paradigms including OOP and functional",
        "Python has extensive libraries for data science and AI",
    ],
    "machine learning": [
        "Machine learning is a subset of artificial intelligence",
        "ML algorithms learn patterns from data",
        "Popular ML frameworks include TensorFlow and PyTorch",
    ],
}


@tool_agent.tool
async def get_weather(
    ctx: RunContext[ToolDependencies],
    location: str,
) -> WeatherResult:
    """
    Get current weather for a location.

    Args:
        ctx: Run context with dependencies.
        location: City name to get weather for.

    Returns:
        WeatherResult with temperature, conditions, and humidity.
    """
    weather_data = ctx.deps.weather_data or DEFAULT_WEATHER_DATA
    location_lower = location.lower()

    if location_lower in weather_data:
        data = weather_data[location_lower]
        return WeatherResult(
            location=location,
            temperature=data["temperature"],
            conditions=data["conditions"],
            humidity=data["humidity"],
        )

    # Default response for unknown locations
    return WeatherResult(
        location=location,
        temperature=20.0,
        conditions="Weather data unavailable",
        humidity=50,
    )


@tool_agent.tool_plain
def calculate(expression: str) -> CalculationResult:
    """
    Perform a mathematical calculation.

    Args:
        expression: A mathematical expression to evaluate.

    Returns:
        CalculationResult with the computed value.
    """
    # Safe evaluation of mathematical expressions
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sum": sum,
    }

    try:
        # Only allow safe mathematical operations
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return CalculationResult(
            expression=expression,
            result=float(result),
            formatted=f"{expression} = {result}",
        )
    except Exception as e:
        return CalculationResult(
            expression=expression,
            result=0.0,
            formatted=f"Error evaluating '{expression}': {e}",
        )


@tool_agent.tool
async def search_information(
    ctx: RunContext[ToolDependencies],
    query: str,
) -> SearchResult:
    """
    Search for information on a topic.

    Args:
        ctx: Run context with dependencies.
        query: The search query.

    Returns:
        SearchResult with relevant information.
    """
    search_data = ctx.deps.search_data or DEFAULT_SEARCH_DATA
    query_lower = query.lower()

    # Find matching results
    results = []
    for key, values in search_data.items():
        if any(word in query_lower for word in key.split()):
            results.extend(values)

    if not results:
        results = [f"No specific information found for '{query}'"]

    return SearchResult(
        query=query,
        results=results[:5],  # Limit to 5 results
        source_count=len(results),
    )


@tool_agent.tool_plain
def get_current_time() -> str:
    """
    Get the current date and time.

    Returns:
        Current datetime as a formatted string.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --8<-- [end:tools]


# --8<-- [start:run]
async def run_tool_agent(
    query: str,
    deps: ToolDependencies | None = None,
) -> ToolResponse:
    """
    Run the tool-using agent with a user query.

    Args:
        query: The user's question or request.
        deps: Optional dependencies for tool execution.

    Returns:
        ToolResponse with the answer and tool usage info.
    """
    if deps is None:
        deps = ToolDependencies()

    print(f"Tool Agent: Processing query '{query}'...")

    result = await tool_agent.run(query, deps=deps)

    print(f"  Tools used: {result.output.tools_used}")
    print(f"  Answer: {result.output.answer[:100]}...")

    return result.output


# Standalone tool functions for direct use
async def standalone_weather(location: str) -> WeatherResult:
    """
    Get weather without agent context (for testing/direct use).

    Args:
        location: City name to get weather for.

    Returns:
        WeatherResult with weather data.
    """
    location_lower = location.lower()
    if location_lower in DEFAULT_WEATHER_DATA:
        data = DEFAULT_WEATHER_DATA[location_lower]
        return WeatherResult(
            location=location,
            temperature=data["temperature"],
            conditions=data["conditions"],
            humidity=data["humidity"],
        )
    return WeatherResult(
        location=location,
        temperature=20.0,
        conditions="Weather data unavailable",
        humidity=50,
    )


def standalone_calculate(expression: str) -> CalculationResult:
    """
    Calculate without agent context (for testing/direct use).

    Args:
        expression: Mathematical expression to evaluate.

    Returns:
        CalculationResult with computed value.
    """
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sum": sum,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return CalculationResult(
            expression=expression,
            result=float(result),
            formatted=f"{expression} = {result}",
        )
    except Exception as e:
        return CalculationResult(
            expression=expression,
            result=0.0,
            formatted=f"Error: {e}",
        )


# --8<-- [end:run]


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        print("=" * 60)
        print("DEMO: Tool Use Pattern")
        print("=" * 60)

        # Demo 1: Weather query
        print("\n--- Weather Query ---")
        result1 = await run_tool_agent("What's the weather like in London?")
        print(f"Full response: {result1.answer}")

        # Demo 2: Calculation
        print("\n--- Calculation Query ---")
        result2 = await run_tool_agent("What is 15 * 7 + 23?")
        print(f"Full response: {result2.answer}")

        # Demo 3: Search
        print("\n--- Search Query ---")
        result3 = await run_tool_agent("Tell me about Python programming")
        print(f"Full response: {result3.answer}")

        # Demo 4: Multi-tool query
        print("\n--- Multi-Tool Query ---")
        result4 = await run_tool_agent(
            "What time is it, and what's the weather in Tokyo?"
        )
        print(f"Full response: {result4.answer}")

    asyncio.run(main())
