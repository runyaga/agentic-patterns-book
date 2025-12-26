# Chapter 9: Learning and Adaptation

Improve performance over time by storing experiences (Few-Shot) and adapting prompts.

## Implementation

Source: `src/agentic_patterns/learning.py`

### Experience Store

Stores input/output pairs to use as future context.

```python
class Experience(BaseModel):
    task_type: str
    input_text: str
    output_text: str
    outcome: str  # "success" or "failure"
    feedback: str

class ExperienceStore:
    def get_relevant_examples(self, task: str, k: int = 3) -> list[Experience]:
        # Filter by task type and success status
        return [e for e in self.experiences if e.task_type == task][-k:]
```

### Idiomatic Pattern (@system_prompt + Deps)

Examples are injected into the system prompt via `@system_prompt` decorator,
keeping the user message clean and focused on the task.

```python
@dataclass
class LearningDeps:
    """Dependencies for learning-enabled agents."""
    store: ExperienceStore
    task_type: str = ""
    use_examples: bool = True
    max_examples: int = 3

# Agent with experience injection
task_agent: Agent[LearningDeps, str] = Agent(
    model,
    system_prompt="Complete the given task using any provided examples.",
    deps_type=LearningDeps,
)

@task_agent.system_prompt
def inject_examples(ctx: RunContext[LearningDeps]) -> str:
    """Inject few-shot examples from experience store into system prompt."""
    if not ctx.deps.use_examples:
        return ""

    examples = ctx.deps.store.get_relevant_examples(
        ctx.deps.task_type, k=ctx.deps.max_examples
    )
    if not examples:
        return ""

    lines = ["Here are successful examples for reference:"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"\nExample {i}:")
        lines.append(f"Input: {ex.input_text[:200]}")
        lines.append(f"Output: {ex.output_text[:200]}")
    return "\n".join(lines)
```

### Execution with Learning

```python
async def run_with_learning(task_type: str, input_text: str, store: ExperienceStore):
    # Create deps - examples injected automatically via @system_prompt
    deps = LearningDeps(store=store, task_type=task_type)

    # Run - user message is just the task
    result = await task_agent.run(input_text, deps=deps)

    return result.output
```

### Feedback Loop

If success rate drops, an optimizer agent can rewrite the prompt:

```python
if feedback_loop.should_adapt(task_type):
    failures = store.get_failure_patterns(task_type)
    new_prompt = await optimizer_agent.run(
        f"Original Prompt: {prompt}\nFailures: {failures}\nFix it."
    )
```

## Use Cases

- **Style Adaptation**: Mimic user's writing style based on past edits.
- **Code Correction**: Learn from linter errors to avoid repeating them.
- **Classification**: Improve label accuracy by providing "gold" examples.

## When to Use

- Tasks are repetitive and belong to distinct categories.
- Quality feedback is available (explicit or implicit).
- Zero-shot performance is insufficient, but Few-shot is promising.

## Example

```bash
.venv/bin/python -m agentic_patterns.learning
```
