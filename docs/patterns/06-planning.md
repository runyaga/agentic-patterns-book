# Chapter 6: Planning

Dynamically decompose goals into steps, execute them, and adapt to failures.

## Implementation

Source: `src/agentic_patterns/planning.py`

### Plan Models

```python
class StepStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

class PlanStep(BaseModel):
    step_number: int
    description: str
    expected_output: str
    dependencies: list[int] = Field(description="Step IDs this depends on")
    status: StepStatus = Field(default=StepStatus.PENDING)

class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]
    reasoning: str
```

### Agents & Execution Loop

```python
planner_agent = Agent(model, output_type=Plan)    # Create initial plan
executor_agent = Agent(model, output_type=StepResult) # Execute one step
replanner_agent = Agent(model, output_type=Plan)  # Fix plan on failure

async def execute_plan(plan: Plan):
    results = []
    for step in plan.steps:
        # Check dependencies
        if not check_deps(step, results): continue

        # Execute
        result = await executor_agent.run(f"Execute {step.description}...")
        results.append(result)

        # Handle Failure (Re-planning)
        if not result.success:
            plan = await replanner_agent.run(
                f"Step failed: {result.error}. Revise plan..."
            )
```

## Use Cases

- **Complex Research**: Decompose "Write report on X" into "Search", "Read", "Outline", "Write".
- **Code Refactoring**: "Analyze file", "Plan changes", "Apply edits", "Run tests".
- **Multi-step Analysis**: "Fetch data", "Clean data", "Run stats", "Visualize".

## When to Use

- Goal is too complex for a single prompt ("one-shot").
- Steps have strict dependencies (B needs output of A).
- Error recovery is needed (if step 2 fails, try step 2b).
- Transparency in process is required.

## Example

```bash
.venv/bin/python -m agentic_patterns.planning
```
