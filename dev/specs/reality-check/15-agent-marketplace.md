# Production Reality Check: Agent Marketplace (The Agora)

**Target file**: `docs/patterns/15-agent-marketplace.md`
**Replaces**: `## Production Reality Check` section (expand existing)

---

## Production Reality Check

### When to Use
- Worker pool is dynamic (agents join/leave at runtime, plugins loaded dynamically)
- Cost/latency tradeoffs vary per request (cheap fast agent vs. expensive thorough one)
- Competitive execution is needed (multiple agents could handle task, pick best)
- Decoupling is valuable (requesters shouldn't know which agents exist)

### When NOT to Use
- Agent pool is static and known at design time (use Router or Supervisor instead)
- Latency is critical (bidding adds round-trip overhead)
- Coordination complexity outweighs benefits for your use case
- Single agent can handle all task types adequately

**Key insight**: For static workflows where you know agents ahead of time, use
**Router (Chapter 2)** or **Supervisor (Chapter 7)**. The Agora adds power but
also latency and complexity.

### Production Considerations
- **Message bus**: Replace `asyncio.Queue` with durable message systems (Redis
  Streams, RabbitMQ, NATS) to handle process crashes.
- **Distributed tracing**: OpenTelemetry/Logfire is critical to visualize the
  async fan-out and debug "why did agent X win?"
- **Economic layer**: Real marketplaces need quota tracking, rate limiting, or
  credits to prevent "Winner's Curse" (agents over-promising).
- **Deadlock prevention**: Circuit breakers for recursive RFPs (agents hiring
  agents hiring agents can spiral).
- **Bid timeout**: Set timeouts for bid collection. Don't wait forever for slow
  agents.
- **Selection strategy**: "Highest confidence" vs "lowest cost" vs "agent
  judgment" have different tradeoffs. Make the strategy configurable.
