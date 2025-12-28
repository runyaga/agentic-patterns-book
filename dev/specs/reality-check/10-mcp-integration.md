# Production Reality Check: MCP Integration

**Target file**: `docs/patterns/10-mcp-integration.md`
**Replaces**: `## When to Use` section

---

## Production Reality Check

### When to Use

| Use Case | Recommended Approach |
| :--- | :--- |
| **Simple Python Tools** | Use standard `@agent.tool`. Easier to debug and deploy. |
| **Shared/Remote Tools** | Use **MCP Integration**. For tools on other servers or in other languages. |
| **Sandboxed Execution** | Use **MCP Integration**. Run MCP server in Docker/WASM for isolation. |

### When NOT to Use
- All tools are Python and run in the same process (standard `@agent.tool` is
  simpler)
- Tool latency is critical (MCP adds IPC overhead vs. direct function calls)
- You don't need tool reuse across languages/agents
- Deployment constraints prevent running separate MCP server processes

### Production Considerations
- **Server lifecycle**: MCP servers are separate processes. Handle startup,
  shutdown, and restarts properly. Consider systemd/supervisord for production.
- **Connection failures**: Network/IPC can fail. Implement reconnection logic
  and graceful degradation when servers are unavailable.
- **Tool discovery caching**: `list_tools()` can be cached to avoid repeated
  discovery calls. Invalidate cache when servers restart.
- **Security**: MCP servers execute code. Run in sandboxed environments (Docker,
  WASM) for untrusted operations. Validate all inputs from agents.
- **Observability**: Log MCP tool calls separately from agent logs. Include
  server name, tool name, inputs, outputs, and latency.
- **Versioning**: Tool schemas can change. Coordinate agent and server updates
  to avoid breaking changes.
