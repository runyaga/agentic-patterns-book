"""Tests for the Prompt Chaining Pattern implementation."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns.prompt_chaining import ChainDeps
from agentic_patterns.prompt_chaining import MarketingEmail
from agentic_patterns.prompt_chaining import ResearchSummary
from agentic_patterns.prompt_chaining import Trend
from agentic_patterns.prompt_chaining import TrendAnalysis
from agentic_patterns.prompt_chaining import run_prompt_chain


class TestModels:
    """Test Pydantic model validation."""

    def test_research_summary_valid(self):
        summary = ResearchSummary(
            key_findings=["Finding 1", "Finding 2"],
            main_themes=["Theme A", "Theme B"],
            market_size="$1.2 trillion",
        )
        assert len(summary.key_findings) == 2
        assert summary.market_size == "$1.2 trillion"

    def test_research_summary_optional_market_size(self):
        summary = ResearchSummary(
            key_findings=["Finding 1"],
            main_themes=["Theme A"],
        )
        assert summary.market_size is None

    def test_trend_valid(self):
        trend = Trend(
            name="AI Integration",
            description="AI is being integrated into consumer devices",
            supporting_data=["67% of smartphones have AI", "45% awareness"],
        )
        assert trend.name == "AI Integration"
        assert len(trend.supporting_data) == 2

    def test_trend_analysis_valid(self):
        trends = TrendAnalysis(
            trends=[
                Trend(
                    name="Trend 1",
                    description="Description 1",
                    supporting_data=["Data point 1"],
                ),
                Trend(
                    name="Trend 2",
                    description="Description 2",
                    supporting_data=["Data point 2"],
                ),
            ]
        )
        assert len(trends.trends) == 2

    def test_marketing_email_valid(self):
        email = MarketingEmail(
            subject="Q4 Market Trends",
            greeting="Hi Team,",
            body="Here are the key trends...",
            call_to_action="Please review and provide feedback.",
            closing="Best regards",
        )
        assert email.subject == "Q4 Market Trends"
        assert "Team" in email.greeting


class TestPromptChain:
    """Test the prompt chaining flow with mocked agents."""

    @pytest.fixture
    def mock_summary(self):
        return ResearchSummary(
            key_findings=[
                "AI adoption increased 67%",
                "Sustainability is a key driver",
                "Wearables grew 28% YoY",
            ],
            main_themes=["AI Integration", "Sustainability", "Health Tech"],
            market_size="$1.2 trillion",
        )

    @pytest.fixture
    def mock_trends(self):
        return TrendAnalysis(
            trends=[
                Trend(
                    name="AI-Powered Devices",
                    description="Consumer electronics feature AI",
                    supporting_data=["67% of smartphones have on-device AI"],
                ),
                Trend(
                    name="Sustainable Tech",
                    description="Eco-friendly products gaining share",
                    supporting_data=["58% prefer eco-friendly"],
                ),
                Trend(
                    name="Health Wearables",
                    description="Health monitoring drives purchases",
                    supporting_data=["72% buy for health features"],
                ),
            ]
        )

    @pytest.fixture
    def mock_email(self):
        return MarketingEmail(
            subject="Key Market Trends for Q4 2024",
            greeting="Hi Marketing Team,",
            body="Based on our latest research, three trends stand out...",
            call_to_action="Please review for our Q1 campaign planning.",
            closing="Best regards,\nResearch Team",
        )

    @pytest.mark.asyncio
    async def test_run_prompt_chain_success(
        self, mock_summary, mock_trends, mock_email
    ):
        """Test full chain execution with mocked agents."""
        mock_summary_result = MagicMock()
        mock_summary_result.output = mock_summary

        mock_trends_result = MagicMock()
        mock_trends_result.output = mock_trends

        mock_email_result = MagicMock()
        mock_email_result.output = mock_email

        with (
            patch(
                "agentic_patterns.prompt_chaining.summarizer_agent"
            ) as mock_summarizer,
            patch(
                "agentic_patterns.prompt_chaining.trend_analyzer_agent"
            ) as mock_analyzer,
            patch(
                "agentic_patterns.prompt_chaining.email_drafter_agent"
            ) as mock_drafter,
        ):
            mock_summarizer.run = AsyncMock(return_value=mock_summary_result)
            mock_analyzer.run = AsyncMock(return_value=mock_trends_result)
            mock_drafter.run = AsyncMock(return_value=mock_email_result)

            result = await run_prompt_chain("Sample market research text")

            assert isinstance(result, MarketingEmail)
            assert result.subject == "Key Market Trends for Q4 2024"

            # Check that deps were passed correctly
            mock_summarizer.run.assert_called_once()
            _, kwargs = mock_summarizer.run.call_args
            assert isinstance(kwargs["deps"], ChainDeps)
            assert kwargs["deps"].raw_text == "Sample market research text"

    @pytest.mark.asyncio
    async def test_chain_passes_data_between_steps(
        self, mock_summary, mock_trends, mock_email
    ):
        """Verify that data flows correctly between chain steps via deps."""
        mock_summary_result = MagicMock()
        mock_summary_result.output = mock_summary

        mock_trends_result = MagicMock()
        mock_trends_result.output = mock_trends

        mock_email_result = MagicMock()
        mock_email_result.output = mock_email

        with (
            patch(
                "agentic_patterns.prompt_chaining.summarizer_agent"
            ) as mock_summarizer,
            patch(
                "agentic_patterns.prompt_chaining.trend_analyzer_agent"
            ) as mock_analyzer,
            patch(
                "agentic_patterns.prompt_chaining.email_drafter_agent"
            ) as mock_drafter,
        ):
            mock_summarizer.run = AsyncMock(return_value=mock_summary_result)
            mock_analyzer.run = AsyncMock(return_value=mock_trends_result)
            mock_drafter.run = AsyncMock(return_value=mock_email_result)

            await run_prompt_chain("Test input")

            # Verify trend analyzer received summary data via DEPS
            # Args: (prompt, deps=...)
            _, analyzer_kwargs = mock_analyzer.run.call_args
            deps = analyzer_kwargs["deps"]
            assert isinstance(deps, ChainDeps)
            assert deps.summary == mock_summary
            # Verify specific data point in the deps object
            assert "AI adoption increased 67%" in deps.summary.key_findings

            # Verify email drafter received trend data via DEPS
            _, drafter_kwargs = mock_drafter.run.call_args
            deps = drafter_kwargs["deps"]
            assert isinstance(deps, ChainDeps)
            assert deps.trends == mock_trends
            assert deps.trends.trends[0].name == "AI-Powered Devices"


class TestModelValidation:
    """Test model validation edge cases."""

    def test_research_summary_empty_lists(self):
        summary = ResearchSummary(
            key_findings=[],
            main_themes=[],
        )
        assert len(summary.key_findings) == 0

    def test_trend_analysis_empty_trends(self):
        analysis = TrendAnalysis(trends=[])
        assert len(analysis.trends) == 0

    def test_trend_with_empty_supporting_data(self):
        trend = Trend(
            name="Test Trend",
            description="A test",
            supporting_data=[],
        )
        assert len(trend.supporting_data) == 0
