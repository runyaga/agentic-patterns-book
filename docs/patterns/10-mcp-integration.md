# MCP Integration (Tool Extensibility)

Source: [`src/agentic_patterns/mcp_integration.py`](https://github.com/runyaga/agentic-patterns-book/blob/main/src/agentic_patterns/mcp_integration.py)

The **MCP Integration** pattern demonstrates how to use the Model Context Protocol (MCP) to extend agents with external tools. Instead of hardcoding tools, this pattern allows agents to connect to local or remote servers (via stdio or HTTP/SSE) to dynamically discover capabilities.

## Key Concepts

```mermaid
sequenceDiagram
    participant Agent
    participant Calc as Calculator Server
    participant FS as Filesystem Server

    Note over Agent: Toolsets=[Calc, FS]
    Agent->>Calc: list_tools()
    Agent->>FS: list_tools()
    Calc-->>Agent: [calc_add, calc_sub]
    FS-->>Agent: [fs_read, fs_list]

    Agent->>Agent: LLM chooses 'calc_add'
    Agent->>Calc: call_tool('calc_add', args, meta={user_id: 123})
    Calc-->>Agent: Result: 42
```

-   **Native Support**: `pydantic-ai` has first-class support for MCP via the `toolsets` parameter. No custom connector code is required.
-   **Multi-Server Orchestration**: Agents can connect to multiple servers simultaneously (e.g., a Calculator server AND a Filesystem server).
-   **Tool Prefixes**: To avoid naming conflicts (e.g., two servers both having a `search` tool), tools can be namespaced (e.g., `calc_add`, `fs_read`).
-   **Dependency Propagation**: Agent-level dependencies (like `user_id` or `auth_token`) can be securely passed down to MCP tool handlers via the `process_tool_call` hook.

## Implementation

This implementation focuses on "Idiomatic pydantic-ai" usage, avoiding unnecessary abstraction layers.

### Creating an MCP Agent

Connecting to a server is as simple as defining an `MCPServerStdio` and passing it to the `Agent`.

```python
--8<-- "src/agentic_patterns/mcp_integration.py:agent"
```

### Multi-Server Support

You can mix and match servers. The LLM sees all available tools and chooses the right one.

```python
--8<-- "src/agentic_patterns/mcp_integration.py:run"
```

### Dependency Injection

Use `process_tool_call` to inject context into every tool invocation.

```python
--8<-- "src/agentic_patterns/mcp_integration.py:deps"
```

## Use Cases

1.  **Tool Reuse**: Write a tool once (e.g., "Database Access") and share it across multiple agents (Python, TypeScript, Go).
2.  **Security Boundaries**: Run tools in a separate process or container (via Docker) to sandbox potentially dangerous operations like file access.
3.  **Local Dev Tools**: Give your agent access to your local CLI tools (git, grep, ls) via stdio servers.

## When to Use

| Use Case | Recommended Approach |
| :--- | :--- |
| **Simple Python Tools** | Use standard `@agent.tool`. Easier to debug and deploy if everything is in one process. |
| **Shared/Remote Tools** | Use **MCP Integration**. Perfect for tools that live on another server or are written in another language. |
| **Sandboxed Execution** | Use **MCP Integration**. Run the MCP server in a restricted environment (Docker/WASM). |

## Example

```bash
# Run the included demo
.venv/bin/python -m agentic_patterns.mcp_integration
```
