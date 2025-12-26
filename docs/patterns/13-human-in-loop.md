# Chapter 13: Human-in-the-Loop

Integrate human oversight for high-stakes decisions or low-confidence outputs.

## Implementation

Source: [`src/agentic_patterns/human_in_loop.py`](https://github.com/runyaga/agentic-patterns-book/blob/main/src/agentic_patterns/human_in_loop.py)

### Data Models & Policies

```python
--8<-- "src/agentic_patterns/human_in_loop.py:models"
```

### Agents

```python
--8<-- "src/agentic_patterns/human_in_loop.py:agents"
```

### Workflow with Oversight

```python
--8<-- "src/agentic_patterns/human_in_loop.py:workflow"
```

## Use Cases

- **Content Moderation**: Review flagged toxic/sensitive content.
- **Financial Ops**: Approve transactions > $10k.
- **Code Deployment**: Agent scaffolds code, Human reviews PR.
- **Medical/Legal**: AI drafts, Expert validates.

## When to Use

- Cost of error is high (financial loss, safety risk).
- Compliance requires human sign-off.
- AI is prone to "hallucination" in the domain.
- Confidence scores are reliable indicators of quality.

## Example

```bash
.venv/bin/python -m agentic_patterns.human_in_loop
```
