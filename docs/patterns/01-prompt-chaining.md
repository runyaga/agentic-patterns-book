# Chapter 1: Prompt Chaining

Chain multiple LLM calls where each step's output becomes the next input.

## Implementation

Source: `src/agentic_patterns/prompt_chaining.py`

### Data Models & Agents

```python
class ResearchSummary(BaseModel):
    key_findings: list[str] = Field(description="List of key findings")
    main_themes: list[str] = Field(description="Main themes identified")
    market_size: str | None = Field(description="Market size if mentioned")

class Trend(BaseModel):
    name: str = Field(description="Name of the trend")
    description: str = Field(description="Brief description")
    supporting_data: list[str] = Field(description="Data points supporting trend")

class TrendAnalysis(BaseModel):
    trends: list[Trend] = Field(description="Top 3 emerging trends")

class MarketingEmail(BaseModel):
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Main email body with trends summary")
    # ... other fields

# Specialized Agents
summarizer_agent = Agent(model, output_type=ResearchSummary, system_prompt="...")
trend_analyzer_agent = Agent(model, output_type=TrendAnalysis, system_prompt="...")
email_drafter_agent = Agent(model, output_type=MarketingEmail, system_prompt="...")
```

### Chain Execution

```python
async def run_prompt_chain(market_research_text: str) -> MarketingEmail:
    # Step 1: Summarize raw research
    summary_result = await summarizer_agent.run(
        f"Summarize the following report:\n\n{market_research_text}"
    )
    summary = summary_result.output

    # Step 2: Identify trends (using summary)
    findings = "\n".join(f"- {f}" for f in summary.key_findings)
    trend_result = await trend_analyzer_agent.run(
        f"Identify top 3 trends from these findings:\n{findings}"
    )
    trends = trend_result.output

    # Step 3: Draft email (using trends)
    trend_details = "\n".join(f"Trend: {t.name}..." for t in trends.trends)
    email_result = await email_drafter_agent.run(
        f"Draft an email about these trends:\n\n{trend_details}"
    )
    return email_result.output
```

## Use Cases

- **Document Processing**: Extract -> Analyze -> Generate
- **Research Pipelines**: Gather -> Synthesize -> Report
- **Content Creation**: Outline -> Draft -> Refine
- **Data Transformation**: Parse -> Process -> Format

## When to Use

- Tasks naturally decompose into sequential steps
- Intermediate results need validation or structured formatting
- Complex reasoning benefits from breaking down the problem

## Example

```bash
.venv/bin/python -m agentic_patterns.prompt_chaining
```
