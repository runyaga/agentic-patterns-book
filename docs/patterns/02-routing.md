# Chapter 2: Routing

Classify user intent and route queries to specialized handlers.

## Implementation

Source: `src/agentic_patterns/routing.py`

### Intent & Router

```python
class Intent(str, Enum):
    ORDER_STATUS = "order_status"
    PRODUCT_INFO = "product_info"
    TECHNICAL_SUPPORT = "technical_support"
    CLARIFICATION = "clarification"

class RouteDecision(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(description="Why this intent was chosen")

# Router Agent
router_agent = Agent(
    model, 
    output_type=RouteDecision,
    system_prompt="Classify intent. If confidence < 0.7, use 'clarification'."
)
```

### Routing Logic

```python
INTENT_HANDLERS = {
    Intent.ORDER_STATUS: order_status_agent,
    Intent.PRODUCT_INFO: product_info_agent,
    # ... other handlers
}

async def route_query(user_query: str) -> tuple[RouteDecision, RouteResponse]:
    # Step 1: Classify intent
    route_result = await router_agent.run(
        f"Classify this query:\n\n{user_query}"
    )
    decision = route_result.output

    # Step 2: Route to handler
    handler = INTENT_HANDLERS[decision.intent]
    handler_result = await handler.run(user_query)

    return decision, handler_result.output
```

## Use Cases

- **Customer Service**: Route to order/product/support teams
- **Multi-domain Q&A**: Route to domain experts
- **Workflow Automation**: Direct tasks to processors
- **Content Moderation**: Route based on content type

## When to Use

- Multiple specialized handlers exist
- Domain expertise varies significantly by query type
- Fallback handling is needed for ambiguous requests (using confidence scores)

## Example

```bash
.venv/bin/python -m agentic_patterns.routing
```
