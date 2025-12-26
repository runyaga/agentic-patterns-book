# Chapter 4: Reflection

Self-correction via iterative evaluation (Producer -> Critic -> Refiner).

## Implementation

Source: `src/agentic_patterns/reflection.py`

### Models & Agents

```python
class Critique(BaseModel):
    is_acceptable: bool = Field(description="Meets quality standards")
    score: float = Field(ge=0.0, le=10.0, description="Quality score (0-10)")
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]

# Producer: Generates initial content
producer_agent = Agent(model, output_type=ProducerOutput, system_prompt="...")

# Critic: Evaluates content against criteria
critic_agent = Agent(model, output_type=Critique, system_prompt="Score 0-10...")

# Refiner: Improves content based on critique
refiner_agent = Agent(model, output_type=RefinedOutput, system_prompt="...")
```

### Reflection Loop

```python
async def run_reflection(task: str, max_iters: int = 3) -> ReflectionResult:
    # 1. Initial Production
    initial = await producer_agent.run(task)
    content = initial.output.content

    for i in range(max_iters):
        # 2. Critique
        critique = (await critic_agent.run(content)).output

        # 3. Check Convergence
        if critique.is_acceptable or critique.score >= 8.0:
            return ReflectionResult(content=content, converged=True, ...)

        # 4. Refine
        refined = await refiner_agent.run(
            f"Content: {content}\nCritique: {critique.suggestions}"
        )
        content = refined.output.content

    return ReflectionResult(content=content, converged=False, ...)
```

## Use Cases

- **Code Generation**: Generate -> Test/Lint -> Fix Errors.
- **Content Writing**: Draft -> Review (Tone/Clarity) -> Revise.
- **Data Extraction**: Extract -> Verify Schema -> Re-extract if invalid.
- **Translation**: Translate -> Check Nuance -> Adjust.

## When to Use

- Quality > Latency (loops take time).
- Success criteria are objectively measurable or distinctly critique-able.
- First-pass results are often imperfect but improvable.

## Example

```bash
.venv/bin/python -m agentic_patterns.reflection
```
