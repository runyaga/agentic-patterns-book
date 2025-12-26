"""
Prompt Chaining Pattern Implementation (Idiomatic PydanticAI).

Based on the Agentic Design Patterns book Chapter 1:
Chain multiple LLM calls where each step's output becomes the next input.

Key concepts:
- Sequential Processing: Each step completes before the next begins
- Data Flow: Output from step N becomes input for step N+1 via Dependencies
- Structured Outputs: Pydantic models ensure type-safe data flow
"""

from dataclasses import dataclass

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent, RunContext

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


@dataclass
class ChainDeps:
    """Dependencies for passing context between chain steps."""
    
    # We store all potential inputs here.
    # In a real app, you might use specific deps for each agent,
    # but a shared context is common for chains.
    raw_text: str | None = None
    summary: ResearchSummary | None = None
    trends: TrendAnalysis | None = None


# Initialize the model
model = get_model()

# --- Agent 1: Summarizer ---
summarizer_agent = Agent(
    model,
    system_prompt=(
        "You are a market research analyst. Your task is to summarize "
        "key findings from market research reports. Focus on extracting "
        "the most important insights, themes, and any quantitative data."
    ),
    deps_type=ChainDeps,
    output_type=ResearchSummary,
)

@summarizer_agent.system_prompt
def inject_raw_text(ctx: RunContext[ChainDeps]) -> str:
    """Inject the raw text to be summarized."""
    if not ctx.deps.raw_text:
        return "No text provided to summarize."
    return f"Report Text:\n{ctx.deps.raw_text}"


# --- Agent 2: Trend Analyzer ---
trend_analyzer_agent = Agent(
    model,
    system_prompt=(
        "You are a trend analyst. Given a summary of market research, "
        "identify the top 3 emerging trends. For each trend, provide "
        "specific data points from the summary that support it."
    ),
    deps_type=ChainDeps,
    output_type=TrendAnalysis,
)

@trend_analyzer_agent.system_prompt
def inject_summary(ctx: RunContext[ChainDeps]) -> str:
    """Inject the summary for trend analysis."""
    if not ctx.deps.summary:
        return "No summary provided."
    
    s = ctx.deps.summary
    findings = "\n".join(f"- {f}" for f in s.key_findings)
    themes = "\n".join(f"- {t}" for t in s.main_themes)
    
    return (
        f"Research Summary:\n"
        f"Key Findings:\n{findings}\n"
        f"Main Themes:\n{themes}\n"
        f"Market Size: {s.market_size or 'N/A'}"
    )


# --- Agent 3: Email Drafter ---
email_drafter_agent = Agent(
    model,
    system_prompt=(
        "You are a professional business communicator. Draft a concise "
        "email to the marketing team that outlines key market trends. "
        "The email should be professional, actionable, and highlight "
        "the most important insights."
    ),
    deps_type=ChainDeps,
    output_type=MarketingEmail,
)

@email_drafter_agent.system_prompt
def inject_trends(ctx: RunContext[ChainDeps]) -> str:
    """Inject trends for email drafting."""
    if not ctx.deps.trends:
        return "No trends provided."
    
    trend_text = []
    for t in ctx.deps.trends.trends:
        data = ", ".join(t.supporting_data)
        trend_text.append(f"Trend: {t.name}\nDesc: {t.description}\nData: {data}")
    
    return "Identified Trends:\n" + "\n\n".join(trend_text)


async def run_prompt_chain(market_research_text: str) -> MarketingEmail:
    """
    Execute the prompt chain using dependencies to pass state.
    """
    # Step 1: Summarize
    print("Step 1: Summarizing market research...")
    deps = ChainDeps(raw_text=market_research_text)
    
    # We pass a generic instruction; the context is in the system prompt via deps
    summary_result = await summarizer_agent.run(
        "Summarize this report.", deps=deps
    )
    summary = summary_result.output
    print(f"  Found {len(summary.key_findings)} key findings")

    # Step 2: Identify trends
    print("Step 2: Identifying trends...")
    deps.summary = summary # Update state
    
    trend_result = await trend_analyzer_agent.run(
        "Identify top 3 trends based on the summary.", deps=deps
    )
    trends = trend_result.output
    print(f"  Identified {len(trends.trends)} trends")

    # Step 3: Draft email
    print("Step 3: Drafting marketing email...")
    deps.trends = trends # Update state
    
    email_result = await email_drafter_agent.run(
        "Draft an email regarding these trends.", deps=deps
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