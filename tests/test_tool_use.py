"""Tests for the Tool Use Pattern implementation."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns.tool_use import CalculationResult
from agentic_patterns.tool_use import SearchResult
from agentic_patterns.tool_use import ToolDependencies
from agentic_patterns.tool_use import ToolResponse
from agentic_patterns.tool_use import WeatherResult
from agentic_patterns.tool_use import run_tool_agent
from agentic_patterns.tool_use import standalone_calculate
from agentic_patterns.tool_use import standalone_weather


class TestModels:
    """Test Pydantic model validation."""

    def test_weather_result_valid(self):
        result = WeatherResult(
            location="London",
            temperature=15.0,
            conditions="Cloudy",
            humidity=78,
        )
        assert result.location == "London"
        assert result.temperature == 15.0
        assert result.humidity == 78

    def test_weather_result_humidity_bounds(self):
        # Valid at boundaries
        WeatherResult(
            location="Test", temperature=0.0, conditions="Cold", humidity=0
        )
        WeatherResult(
            location="Test", temperature=40.0, conditions="Hot", humidity=100
        )

        # Invalid
        with pytest.raises(ValueError):
            WeatherResult(
                location="Test",
                temperature=20.0,
                conditions="Test",
                humidity=101,
            )

        with pytest.raises(ValueError):
            WeatherResult(
                location="Test",
                temperature=20.0,
                conditions="Test",
                humidity=-1,
            )

    def test_calculation_result_valid(self):
        result = CalculationResult(
            expression="2 + 2",
            result=4.0,
            formatted="2 + 2 = 4",
        )
        assert result.expression == "2 + 2"
        assert result.result == 4.0
        assert result.formatted == "2 + 2 = 4"

    def test_search_result_valid(self):
        result = SearchResult(
            query="python",
            results=["Result 1", "Result 2"],
            source_count=2,
        )
        assert result.query == "python"
        assert len(result.results) == 2
        assert result.source_count == 2

    def test_search_result_empty_results(self):
        result = SearchResult(
            query="unknown",
            results=[],
            source_count=0,
        )
        assert result.results == []
        assert result.source_count == 0

    def test_tool_response_valid(self):
        response = ToolResponse(
            answer="The weather in London is 15°C",
            tools_used=["get_weather"],
            reasoning="Used weather tool to get current conditions",
        )
        assert response.answer == "The weather in London is 15°C"
        assert "get_weather" in response.tools_used
        assert response.reasoning != ""

    def test_tool_response_multiple_tools(self):
        response = ToolResponse(
            answer="Combined answer",
            tools_used=["get_weather", "calculate", "search_information"],
            reasoning="Used multiple tools",
        )
        assert len(response.tools_used) == 3

    def test_tool_response_no_tools(self):
        response = ToolResponse(
            answer="Simple answer without tools",
            tools_used=[],
            reasoning="No tools needed",
        )
        assert len(response.tools_used) == 0


class TestToolDependencies:
    """Test ToolDependencies configuration."""

    def test_dependencies_default(self):
        deps = ToolDependencies()
        assert deps.weather_data is None
        assert deps.search_data is None

    def test_dependencies_custom_weather(self):
        custom_weather = {
            "berlin": {
                "temperature": 10.0,
                "conditions": "Snowy",
                "humidity": 80,
            }
        }
        deps = ToolDependencies(weather_data=custom_weather)
        assert "berlin" in deps.weather_data
        assert deps.weather_data["berlin"]["temperature"] == 10.0

    def test_dependencies_custom_search(self):
        custom_search = {
            "ai": ["Artificial Intelligence info"],
        }
        deps = ToolDependencies(search_data=custom_search)
        assert "ai" in deps.search_data


class TestStandaloneWeather:
    """Test standalone weather function."""

    @pytest.mark.asyncio
    async def test_known_location(self):
        result = await standalone_weather("London")
        assert result.location == "London"
        assert result.temperature == 15.0
        assert "rain" in result.conditions.lower()
        assert result.humidity == 78

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        result = await standalone_weather("TOKYO")
        assert result.location == "TOKYO"
        assert result.temperature == 22.0
        assert result.conditions == "Clear skies"

    @pytest.mark.asyncio
    async def test_unknown_location(self):
        result = await standalone_weather("Unknown City")
        assert result.location == "Unknown City"
        assert result.temperature == 20.0
        assert "unavailable" in result.conditions.lower()


class TestStandaloneCalculate:
    """Test standalone calculate function."""

    def test_simple_addition(self):
        result = standalone_calculate("2 + 2")
        assert result.result == 4.0
        assert "2 + 2" in result.formatted

    def test_multiplication(self):
        result = standalone_calculate("15 * 7")
        assert result.result == 105.0

    def test_complex_expression(self):
        result = standalone_calculate("(10 + 5) * 2")
        assert result.result == 30.0

    def test_with_allowed_functions(self):
        result = standalone_calculate("abs(-5)")
        assert result.result == 5.0

        result = standalone_calculate("max(1, 5, 3)")
        assert result.result == 5.0

        result = standalone_calculate("pow(2, 3)")
        assert result.result == 8.0

    def test_division(self):
        result = standalone_calculate("100 / 4")
        assert result.result == 25.0

    def test_invalid_expression(self):
        result = standalone_calculate("invalid expression")
        assert result.result == 0.0
        assert "Error" in result.formatted

    def test_dangerous_expression_blocked(self):
        # Should fail because __import__ is not allowed
        result = standalone_calculate("__import__('os').system('ls')")
        assert result.result == 0.0
        assert "Error" in result.formatted


class TestRunToolAgent:
    """Test the tool-using agent integration."""

    @pytest.fixture
    def mock_tool_response_weather(self):
        return ToolResponse(
            answer="The weather in London is 15°C with cloudy conditions.",
            tools_used=["get_weather"],
            reasoning="Used weather tool to check London weather.",
        )

    @pytest.fixture
    def mock_tool_response_calculate(self):
        return ToolResponse(
            answer="15 * 7 + 23 = 128",
            tools_used=["calculate"],
            reasoning="Used calculator tool to compute the expression.",
        )

    @pytest.fixture
    def mock_tool_response_multi(self):
        return ToolResponse(
            answer="It's 3:30 PM and Tokyo is 22°C with clear skies.",
            tools_used=["get_current_time", "get_weather"],
            reasoning="Used time and weather tools.",
        )

    @pytest.mark.asyncio
    async def test_weather_query(self, mock_tool_response_weather):
        """Test agent handles weather queries."""
        mock_result = MagicMock()
        mock_result.output = mock_tool_response_weather

        with patch("agentic_patterns.tool_use.tool_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            response = await run_tool_agent("What's the weather in London?")

            assert isinstance(response, ToolResponse)
            assert "get_weather" in response.tools_used
            assert "London" in response.answer

    @pytest.mark.asyncio
    async def test_calculation_query(self, mock_tool_response_calculate):
        """Test agent handles calculation queries."""
        mock_result = MagicMock()
        mock_result.output = mock_tool_response_calculate

        with patch("agentic_patterns.tool_use.tool_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            response = await run_tool_agent("What is 15 * 7 + 23?")

            assert "calculate" in response.tools_used
            assert "128" in response.answer

    @pytest.mark.asyncio
    async def test_multi_tool_query(self, mock_tool_response_multi):
        """Test agent handles queries requiring multiple tools."""
        mock_result = MagicMock()
        mock_result.output = mock_tool_response_multi

        with patch("agentic_patterns.tool_use.tool_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            response = await run_tool_agent(
                "What time is it and what's the weather in Tokyo?"
            )

            assert len(response.tools_used) == 2
            assert "get_current_time" in response.tools_used
            assert "get_weather" in response.tools_used

    @pytest.mark.asyncio
    async def test_custom_dependencies(self, mock_tool_response_weather):
        """Test agent accepts custom dependencies."""
        mock_result = MagicMock()
        mock_result.output = mock_tool_response_weather

        custom_deps = ToolDependencies(
            weather_data={
                "berlin": {
                    "temperature": 5.0,
                    "conditions": "Snowy",
                    "humidity": 85,
                }
            }
        )

        with patch("agentic_patterns.tool_use.tool_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            _response = await run_tool_agent(
                "What's the weather in Berlin?",
                deps=custom_deps,
            )

            # Verify agent was called with custom deps
            call_args = mock_agent.run.call_args
            assert call_args[1]["deps"] == custom_deps

    @pytest.mark.asyncio
    async def test_default_dependencies(self, mock_tool_response_weather):
        """Test agent uses default dependencies when none provided."""
        mock_result = MagicMock()
        mock_result.output = mock_tool_response_weather

        with patch("agentic_patterns.tool_use.tool_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            await run_tool_agent("What's the weather?")

            call_args = mock_agent.run.call_args
            deps = call_args[1]["deps"]
            assert isinstance(deps, ToolDependencies)
            assert deps.weather_data is None
            assert deps.search_data is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_weather_result_negative_temperature(self):
        result = WeatherResult(
            location="Antarctica",
            temperature=-40.0,
            conditions="Freezing",
            humidity=30,
        )
        assert result.temperature == -40.0

    def test_calculation_result_zero(self):
        result = CalculationResult(
            expression="0 * 100",
            result=0.0,
            formatted="0 * 100 = 0",
        )
        assert result.result == 0.0

    def test_calculation_result_large_number(self):
        result = CalculationResult(
            expression="10 ** 10",
            result=10000000000.0,
            formatted="10 ** 10 = 10000000000",
        )
        assert result.result == 10000000000.0

    def test_search_result_many_results(self):
        results = [f"Result {i}" for i in range(100)]
        result = SearchResult(
            query="test",
            results=results,
            source_count=100,
        )
        assert len(result.results) == 100
        assert result.source_count == 100

    def test_tool_response_long_answer(self):
        long_answer = "A" * 1000
        response = ToolResponse(
            answer=long_answer,
            tools_used=["search_information"],
            reasoning="Long answer needed",
        )
        assert len(response.answer) == 1000

    def test_standalone_calculate_float_result(self):
        result = standalone_calculate("10 / 3")
        assert abs(result.result - 3.3333333) < 0.0001

    def test_standalone_calculate_nested_functions(self):
        result = standalone_calculate("max(abs(-5), min(10, 20))")
        assert result.result == 10.0


class TestDecoratedToolFunctions:
    """Test the @tool decorated functions directly with mock context."""

    @pytest.fixture
    def mock_context_default_deps(self):
        """Create a mock RunContext with default dependencies."""
        ctx = MagicMock()
        ctx.deps = ToolDependencies()
        return ctx

    @pytest.fixture
    def mock_context_custom_weather(self):
        """Create a mock RunContext with custom weather data."""
        ctx = MagicMock()
        ctx.deps = ToolDependencies(
            weather_data={
                "berlin": {
                    "temperature": 5.0,
                    "conditions": "Snowy",
                    "humidity": 85,
                }
            }
        )
        return ctx

    @pytest.fixture
    def mock_context_custom_search(self):
        """Create a mock RunContext with custom search data."""
        ctx = MagicMock()
        ctx.deps = ToolDependencies(
            search_data={
                "artificial intelligence": [
                    "AI is transforming industries",
                    "Machine learning is a subset of AI",
                ],
            }
        )
        return ctx

    @pytest.mark.asyncio
    async def test_get_weather_known_location(self, mock_context_default_deps):
        """Test get_weather with known location from default data."""
        from agentic_patterns.tool_use import get_weather

        result = await get_weather(mock_context_default_deps, "London")
        assert result.location == "London"
        assert result.temperature == 15.0
        assert result.humidity == 78

    @pytest.mark.asyncio
    async def test_get_weather_unknown(self, mock_context_default_deps):
        """Test get_weather with unknown location."""
        from agentic_patterns.tool_use import get_weather

        result = await get_weather(mock_context_default_deps, "Unknown City")
        assert result.location == "Unknown City"
        assert result.temperature == 20.0
        assert "unavailable" in result.conditions.lower()

    @pytest.mark.asyncio
    async def test_get_weather_custom_data(self, mock_context_custom_weather):
        """Test get_weather with custom weather data."""
        from agentic_patterns.tool_use import get_weather

        result = await get_weather(mock_context_custom_weather, "Berlin")
        assert result.location == "Berlin"
        assert result.temperature == 5.0
        assert result.conditions == "Snowy"
        assert result.humidity == 85

    def test_calculate_simple(self):
        """Test calculate tool with simple expression."""
        from agentic_patterns.tool_use import calculate

        result = calculate("2 + 2")
        assert result.result == 4.0
        assert "4" in result.formatted

    def test_calculate_complex(self):
        """Test calculate tool with complex expression."""
        from agentic_patterns.tool_use import calculate

        result = calculate("(10 + 5) * 2 - 3")
        assert result.result == 27.0

    def test_calculate_error_handling(self):
        """Test calculate tool with invalid expression."""
        from agentic_patterns.tool_use import calculate

        result = calculate("invalid syntax here")
        assert result.result == 0.0
        assert "Error" in result.formatted

    @pytest.mark.asyncio
    async def test_search_information_match(self, mock_context_default_deps):
        """Test search_information with matching query."""
        from agentic_patterns.tool_use import search_information

        result = await search_information(
            mock_context_default_deps, "python programming"
        )
        assert result.query == "python programming"
        assert len(result.results) > 0
        assert result.source_count > 0

    @pytest.mark.asyncio
    async def test_search_no_match(self, mock_context_default_deps):
        """Test search_information with non-matching query."""
        from agentic_patterns.tool_use import search_information

        result = await search_information(
            mock_context_default_deps, "xyz nonexistent topic"
        )
        assert "No specific information found" in result.results[0]

    @pytest.mark.asyncio
    async def test_search_custom_data(self, mock_context_custom_search):
        """Test search_information with custom search data."""
        from agentic_patterns.tool_use import search_information

        result = await search_information(
            mock_context_custom_search, "artificial intelligence applications"
        )
        assert len(result.results) > 0
        assert any("AI" in r for r in result.results)

    def test_get_current_time_format(self):
        """Test get_current_time returns properly formatted string."""
        from agentic_patterns.tool_use import get_current_time

        result = get_current_time()
        # Should be in format YYYY-MM-DD HH:MM:SS
        assert len(result) == 19
        assert result[4] == "-"
        assert result[7] == "-"
        assert result[10] == " "
        assert result[13] == ":"
        assert result[16] == ":"
