# Chapter 1: Prompt Chaining

Chain multiple LLM calls where each step's output becomes the next input.

## Implementation

Source: `src/agentic_patterns/prompt_chaining.py`

### Data Models & Agents

```python
@dataclass
class ChainDeps:
    """Dependencies for passing context between chain steps."""
    raw_text: str | None = None
    summary: ResearchSummary | None = None
    trends: TrendAnalysis | None = None

class ResearchSummary(BaseModel):
    key_findings: list[str] = Field(description="List of key findings")
    main_themes: list[str] = Field(description="Main themes identified")
    market_size: str | None = Field(description="Market size if mentioned")

# specialized agents with deps_type=ChainDeps
summarizer_agent = Agent(
    model, 
    deps_type=ChainDeps,
    output_type=ResearchSummary, 
    system_prompt="..."
)
# ... trend_analyzer_agent, email_drafter_agent follow same pattern
```

### Chain Execution

```python
async def run_prompt_chain(market_research_text: str) -> MarketingEmail:
    # Step 1: Summarize
    deps = ChainDeps(raw_text=market_research_text)
    summary_result = await summarizer_agent.run(
        "Summarize this report.", deps=deps
    )
    summary = summary_result.output

    # Step 2: Identify trends (pass state via deps)
    deps.summary = summary
    trend_result = await trend_analyzer_agent.run(
        "Identify top 3 trends based on the summary.", deps=deps
    )
    trends = trend_result.output

    # Step 3: Draft email
    deps.trends = trends
    email_result = await email_drafter_agent.run(
        "Draft an email regarding these trends.", deps=deps
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
