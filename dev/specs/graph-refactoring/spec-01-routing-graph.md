# Spec 01: Routing Pattern - Keep Simple Implementation

**Status**: FINAL
**Decision**: KEEP - functional implementation is simpler and sufficient
**Priority**: N/A (no changes needed)
**Complexity**: N/A

---

## 1. Decision Summary

### 1.1 Recommendation: KEEP Original

The existing functional `route_query()` implementation should be **kept as-is**. The dictionary-based dispatch pattern is:

- **Simple**: ~10 lines of core logic
- **Readable**: Immediately understandable
- **Sufficient**: Does exactly what's needed, nothing more

### 1.2 Why NOT Graph

A graph-based implementation was prototyped and rejected because:

| Factor | Functional | Graph |
|--------|-----------|-------|
| Lines of code | ~30 | ~100 |
| Cognitive load | Low | Medium |
| Value added | N/A | Minimal |
| Maintenance | Easy | More complex |

The graph version adds ~70 lines of boilerplate (5 node classes, state dataclass) for benefits that don't justify the complexity in this simple routing case.

---

## 2. Current Implementation (KEEP)

```python
# Simple, elegant dictionary dispatch

INTENT_HANDLERS = {
    Intent.ORDER_STATUS: order_status_agent,
    Intent.PRODUCT_INFO: product_info_agent,
    Intent.TECHNICAL_SUPPORT: technical_support_agent,
    Intent.CLARIFICATION: clarification_agent,
}

async def route_query(user_query: str) -> tuple[RouteDecision, RouteResponse]:
    # Step 1: Classify intent
    route_result = await router_agent.run(classify_prompt)
    decision = route_result.output

    # Step 2: Dispatch via dictionary lookup
    handler = INTENT_HANDLERS[decision.intent]

    # Step 3: Execute handler
    handler_result = await handler.run(user_query)

    return decision, handler_result.output
```

**Why this works well:**

1. **Dictionary dispatch is a Pythonic pattern** - well-understood, efficient
2. **Single responsibility** - classify, lookup, execute
3. **Easy to extend** - add new intent = add enum + handler + dict entry
4. **No framework lock-in** - plain Python

---

## 3. Rejected Graph Implementation

For reference, here's what was prototyped and rejected:

```python
# REJECTED: Over-engineered for this use case

@dataclass
class RoutingState:
    user_query: str
    decision: RouteDecision | None = None

@dataclass
class ClassifyIntentNode(BaseNode[RoutingState, None, RoutingResult]):
    async def run(self, ctx):
        # ... 15 lines ...
        match intent:
            case Intent.ORDER_STATUS: return OrderStatusNode()
            # ... 3 more cases ...

@dataclass
class OrderStatusNode(BaseNode[RoutingState, None, RoutingResult]):
    async def run(self, ctx) -> End[RoutingResult]:
        # ... 5 lines ...

# ... 3 more identical node classes ...

routing_graph = Graph(nodes=[...])  # 5 nodes
```

**Problems:**
- 5 nearly-identical node classes
- State management overhead for a stateless operation
- Graph framework for a non-cyclic, non-branching flow
- ~3x more code for same functionality

---

## 4. When Graph WOULD Make Sense

Graph-based routing would be justified if:

- **Multi-step routing**: Query needs multiple classification stages
- **Stateful routing**: Decisions depend on accumulated context
- **Cyclic flows**: Some routes loop back for refinement
- **Complex branching**: More than simple intent -> handler mapping

The current routing is none of these - it's a simple classify-and-dispatch.

---

## 5. Comparison with Other Patterns

| Pattern | Complexity | Graph Value | Decision |
|---------|-----------|-------------|----------|
| Routing | Simple (dictionary dispatch) | Low | **KEEP** |
| Multi-Agent | Complex (cyclic supervisor loop) | High | REPLACE |
| Human-in-Loop | Complex (approval cycles) | High | NEW |

Routing stays simple. Multi-agent and HITL benefit from graphs.

---

## 6. No Action Required

- No code changes to `routing.py`
- No test changes to `test_routing.py`
- Pattern works well as-is
