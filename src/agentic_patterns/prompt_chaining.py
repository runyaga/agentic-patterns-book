"""
Prompt Chaining Pattern Implementation.

Based on the Agentic Design Patterns book Chapter 1:
Chain multiple LLM calls where each step's output becomes the next input.

Example use case: Market research analysis pipeline
1. Summarize market research findings
2. Identify trends with supporting data
3. Draft an email to the marketing team
"""

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent

from agentic_patterns._models import get_model


class ResearchSummary(BaseModel):
    """Structured summary of market research findings."""

    key_findings: list[str] = Field(
        description="List of key findings from the report"
    )
    main_themes: list[str] = Field(description="Main themes identified")
    market_size: str | None = Field(
        default=None, description="Market size if mentioned"
    )


class Trend(BaseModel):
    """A single identified trend with supporting data."""

    name: str = Field(description="Name of the trend")
    description: str = Field(description="Brief description of the trend")
    supporting_data: list[str] = Field(
        description="Data points that support this trend"
    )


class TrendAnalysis(BaseModel):
    """Analysis of top trends from the summary."""

    trends: list[Trend] = Field(description="Top 3 emerging trends")


class MarketingEmail(BaseModel):
    """Structured email content."""

    subject: str = Field(description="Email subject line")
    greeting: str = Field(description="Email greeting")
    body: str = Field(description="Main email body with trends summary")
    call_to_action: str = Field(description="What you want the team to do")
    closing: str = Field(description="Email closing")


# Initialize the model
model = get_model()

# Define agents for each step in the chain
summarizer_agent = Agent(
    model,
    system_prompt=(
        "You are a market research analyst. Your task is to summarize "
        "key findings from market research reports. Focus on extracting "
        "the most important insights, themes, and any quantitative data."
    ),
    output_type=ResearchSummary,
)

trend_analyzer_agent = Agent(
    model,
    system_prompt=(
        "You are a trend analyst. Given a summary of market research, "
        "identify the top 3 emerging trends. For each trend, provide "
        "specific data points from the summary that support it."
    ),
    output_type=TrendAnalysis,
)

email_drafter_agent = Agent(
    model,
    system_prompt=(
        "You are a professional business communicator. Draft a concise "
        "email to the marketing team that outlines key market trends. "
        "The email should be professional, actionable, and highlight "
        "the most important insights."
    ),
    output_type=MarketingEmail,
)


async def run_prompt_chain(market_research_text: str) -> MarketingEmail:
    """
    Execute the prompt chain.

    Args:
        market_research_text: Raw market research report text.

    Returns:
        MarketingEmail with the final drafted email.

    Steps:
        1. Summarize the research
        2. Analyze trends from the summary
        3. Draft an email based on the trends
    """
    # Step 1: Summarize
    print("Step 1: Summarizing market research...")
    summary_result = await summarizer_agent.run(
        f"Summarize the following market research report:\n\n"
        f"{market_research_text}"
    )
    summary = summary_result.output
    print(f"  Found {len(summary.key_findings)} key findings")

    # Step 2: Identify trends (using summary as input)
    print("Step 2: Identifying trends...")
    findings = "\n".join(f"- {f}" for f in summary.key_findings)
    themes = "\n".join(f"- {t}" for t in summary.main_themes)
    trend_result = await trend_analyzer_agent.run(
        f"Based on this research summary, identify the top 3 trends:\n\n"
        f"Key Findings:\n{findings}\n\n"
        f"Main Themes:\n{themes}\n\n"
        f"Market Size: {summary.market_size or 'Not specified'}"
    )
    trends = trend_result.output
    print(f"  Identified {len(trends.trends)} trends")

    # Step 3: Draft email (using trends as input)
    print("Step 3: Drafting marketing email...")
    trend_details = "\n".join(
        f"Trend: {t.name}\n"
        f"Description: {t.description}\n"
        f"Supporting Data:\n"
        + "\n".join(f"  - {d}" for d in t.supporting_data)
        for t in trends.trends
    )
    email_result = await email_drafter_agent.run(
        f"Draft an email about these market trends:\n\n{trend_details}"
    )

    print("Chain complete!")
    return email_result.output


if __name__ == "__main__":
    import asyncio

    sample_report = """
    Q4 2024 Consumer Electronics Market Research Report

    Executive Summary:
    The global consumer electronics market reached $1.2 trillion in 2024,
    representing a 7.3% year-over-year growth.

    Key Findings:
    1. AI Integration: 67% of new smartphones now feature on-device AI.
    2. Sustainability Focus: 58% of consumers prefer eco-friendly products.
    3. Wearables Growth: Smartwatch market grew 28% YoY.
    4. Smart Home Adoption: 43% of households have smart home devices.
    """

    async def main() -> None:
        email = await run_prompt_chain(sample_report)
        print("\n" + "=" * 60)
        print("FINAL OUTPUT - Marketing Email")
        print("=" * 60)
        print(f"\nSubject: {email.subject}")
        print(f"\n{email.greeting}")
        print(f"\n{email.body}")
        print(f"\n{email.call_to_action}")
        print(f"\n{email.closing}")

    asyncio.run(main())
