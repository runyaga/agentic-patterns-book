# Chapter 4: Reflection

Self-correction via automated validation loops.

## Implementation

Source: `src/agentic_patterns/reflection.py`

### Idiomatic Pattern (ModelRetry)

Instead of a manual `while` loop, we use PydanticAI's `output_validator` (or `result_validator` in older versions) to critique the output. If the quality is insufficient, we raise `ModelRetry`, which automatically feeds the error back to the model for a new attempt.

```python
# 1. Define Dependencies
@dataclass
class ReflectionDeps:
    critic_agent: Agent

# 2. Define Producer with Retry Policy
producer_agent = Agent(
    model,
    deps_type=ReflectionDeps,
    output_type=ProducerOutput,
    retries=3  # Allow 3 attempts
)

# 3. Define the Validator Hook
@producer_agent.output_validator
async def validate_quality(ctx: RunContext[ReflectionDeps], result: ProducerOutput) -> ProducerOutput:
    # Call the critic agent
    critique = await ctx.deps.critic_agent.run(result.content)
    
    # Check thresholds
    if critique.score < 8.0:
        # PydanticAI Magic: This raises an error that the model sees, 
        # prompting it to fix the specific issues mentioned.
        raise ModelRetry(
            f"Score {critique.score}/10. Feedback: {critique.feedback}"
        )
        
    return result
```

### Execution

The caller code is incredibly simple because the loop is internal.

```python
async def run_reflection(task: str):
    # Dependencies needed for the validation hook
    deps = ReflectionDeps(critic_agent=critic_agent)
    
    # Run - this single call handles the generate -> critique -> retry loop
    result = await producer_agent.run(task, deps=deps)
    return result.output
```

## Use Cases

- **Code Generation**: Validator runs unit tests; retries on failure.
- **Content Writing**: Validator checks style guidelines; retries on violations.
- **Data Extraction**: Validator checks schema constraints; retries on mismatches.

## When to Use

- You have a clear "pass/fail" or scoring criteria.
- You want to keep your main application logic linear (`result = run()`) rather than looping.
- You want the model to see *why* it failed (the `ModelRetry` message becomes conversation history).

## Example

```bash
.venv/bin/python -m agentic_patterns.reflection
```