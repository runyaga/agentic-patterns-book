# Chapter 7: Multi-Agent Collaboration

Coordinate specialized agents (Supervisor, Researcher, Writer) to achieve complex goals.

## Implementation

Source: `src/agentic_patterns/multi_agent.py`

### Roles & Plans

```python
class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    # ...

class DelegatedTask(BaseModel):
    task_id: int
    assigned_to: AgentRole
    description: str
    status: TaskStatus = Field(default=TaskStatus.PENDING)

class SupervisorPlan(BaseModel):
    objective: str
    tasks: list[DelegatedTask] = Field(description="Delegation strategy")

# Supervisor Agent
supervisor_agent = Agent(
    model, 
    output_type=SupervisorPlan,
    system_prompt="Coordinate team: Researcher, Analyst, Writer..."
)
```

### Execution Loop (Supervisor Pattern)

```python
async def run_collaborative_task(objective: str):
    context = CollaborationContext(messages=[], task_results=[])

    # 1. Supervisor plans delegation
    plan = await supervisor_agent.run(f"Plan for: {objective}")
    
    # 2. Execute tasks sequentially
    for task in plan.output.tasks:
        worker = ROLE_AGENTS[task.assigned_to]
        
        # Workers can see previous results via tools/context
        result = await worker.run(
            f"Task: {task.description}", 
            deps=context
        )
        context.task_results.append(result.output)

    # 3. Synthesize final output
    return await synthesizer_agent.run(
        f"Combine results for: {objective}", deps=context
    )
```

### Network Pattern (Parallel)

Alternatively, agents work in parallel without a strict supervisor plan:

```python
async def run_network(objective: str, roles: list[AgentRole]):
    # Consult all agents in parallel
    results = await asyncio.gather(*[
        ROLE_AGENTS[role].run(f"Perspective on: {objective}") 
        for role in roles
    ])
    # Synthesize their independent perspectives
    return await synthesizer_agent.run(format_results(results))
```

## Use Cases

- **Content Factory**: Research -> Outline -> Write -> Review.
- **Software Dev**: Architect -> Backend Dev -> Frontend Dev -> QA.
- **Complex Analysis**: Legal Analyst + Financial Analyst + Risk Officer.

## When to Use

- Task exceeds single context window or single agent's capability.
- Distinct specialized skills are required (e.g., coding vs. writing).
- Parallel processing (Network) or sequential validation (Supervisor) is needed.

## Example

```bash
.venv/bin/python -m agentic_patterns.multi_agent
```
