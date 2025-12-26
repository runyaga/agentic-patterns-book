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

### Execution with Learning

```python
async def run_learning_task(task_type: str, user_input: str, store: ExperienceStore):
    # 1. Retrieve successful examples (Few-Shot)
    examples = store.get_relevant_examples(task_type, k=3)
    context = format_examples(examples)

    # 2. Run agent with context
    result = await agent.run(f"{context}\n\nTask: {user_input}")

    # 3. (Optional) Record outcome for future
    store.add_experience(
        task_type=task_type,
        input=user_input, 
        output=result.output,
        outcome="success" # or based on user feedback
    )
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
