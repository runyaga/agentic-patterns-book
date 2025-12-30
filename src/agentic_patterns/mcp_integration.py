"""
MCP Integration Pattern - Idiomatic pydantic-ai usage.

Based on the Agentic Design Patterns book Chapter 10:
Demonstrates native MCP (Model Context Protocol) support in pydantic-ai.

Key concepts:
- toolsets parameter for multi-server support
- tool_prefix for avoiding naming conflicts
- process_tool_call for deps propagation

No custom connector class needed - pydantic-ai handles it all.
"""

import asyncio
import sys
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai import RunContext
from pydantic_ai.mcp import CallToolFunc
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.mcp import ToolResult
from pydantic_ai.models import Model

from agentic_patterns._models import get_model


# --8<-- [start:deps]
@dataclass
class MCPDeps:
    """Dependencies to pass through to MCP tools."""

    user_id: str | None = None
    request_id: str | None = None
    context: dict[str, Any] | None = None


async def process_tool_call_with_deps(
    ctx: RunContext[MCPDeps],
    call_tool: CallToolFunc,
    name: str,
    tool_args: dict[str, Any],
) -> ToolResult:
    """
    Inject deps into MCP tool calls.

    The MCP server can access these via ctx.request_context.meta
    """
    meta: dict[str, Any] = {}
    if ctx.deps.user_id:
        meta["user_id"] = ctx.deps.user_id
    if ctx.deps.request_id:
        meta["request_id"] = ctx.deps.request_id
    if ctx.deps.context:
        meta["context"] = ctx.deps.context
    return await call_tool(name, tool_args, meta if meta else None)


# --8<-- [end:deps]


# --8<-- [start:servers]
def create_calculator_server(
    tool_prefix: str | None = None,
) -> MCPServerStdio:
    """Create calculator MCP server."""
    return MCPServerStdio(
        sys.executable,
        args=["-m", "agentic_patterns.mcp_servers.calculator"],
        tool_prefix=tool_prefix,
    )


def create_filesystem_server(
    tool_prefix: str | None = None,
) -> MCPServerStdio:
    """Create filesystem MCP server."""
    return MCPServerStdio(
        sys.executable,
        args=["-m", "agentic_patterns.mcp_servers.filesystem"],
        tool_prefix=tool_prefix,
    )


# --8<-- [end:servers]


# --8<-- [start:agent]
def create_mcp_agent(
    servers: list[MCPServerStdio] | None = None,
    system_prompt: str = "Use the available tools to help the user.",
    with_deps: bool = False,
    model: Model | None = None,
) -> Agent[MCPDeps, str] | Agent[None, str]:
    """
    Create an agent with MCP server tools.

    This demonstrates idiomatic pydantic-ai MCP usage.
    No custom connector - just native toolsets.

    Args:
        servers: MCP servers to connect. Defaults to calculator.
        system_prompt: Agent system prompt.
        with_deps: Whether to enable deps propagation.
        model: pydantic-ai Model instance. If None, uses default model.

    Returns:
        Configured agent with MCP tools.

    Example:
        agent = create_mcp_agent()
        async with agent:
            result = await agent.run("What is 15 * 7?")
    """
    if servers is None:
        servers = [create_calculator_server()]

    agent_model = model or get_model()

    if with_deps:
        # Recreate servers with deps propagation
        servers_with_deps = []
        for server in servers:
            new_server = MCPServerStdio(
                server.command,
                args=server.args,
                tool_prefix=server.tool_prefix,
                process_tool_call=process_tool_call_with_deps,
            )
            servers_with_deps.append(new_server)

        return Agent(
            agent_model,
            system_prompt=system_prompt,
            deps_type=MCPDeps,
            toolsets=servers_with_deps,
        )

    return Agent(
        agent_model,
        system_prompt=system_prompt,
        toolsets=servers,
    )


# --8<-- [end:agent]


# --8<-- [start:run]
async def run_with_mcp_tools(
    prompt: str,
    servers: list[MCPServerStdio] | None = None,
    deps: MCPDeps | None = None,
    model: Model | None = None,
) -> str:
    """
    Run a prompt with MCP tool access.

    Simple entry point showing idiomatic usage.

    Args:
        prompt: The user prompt.
        servers: MCP servers to use. Defaults to calculator.
        deps: Optional dependencies to pass to tools.
        model: pydantic-ai Model instance. If None, uses default model.

    Returns:
        Agent response as string.

    Example:
        result = await run_with_mcp_tools("What is 15 * 7?")
        print(result)  # "The result of 15 * 7 is 105."
    """
    with_deps = deps is not None
    agent = create_mcp_agent(servers=servers, with_deps=with_deps, model=model)

    async with agent:
        if deps:
            result = await agent.run(prompt, deps=deps)
        else:
            result = await agent.run(prompt)
        return result.output


async def run_multi_server_example(model: Model | None = None) -> str:
    """
    Example: Agent with multiple MCP servers.

    Shows how to combine calculator and filesystem tools.

    Args:
        model: pydantic-ai Model instance. If None, uses default model.
    """
    calc = create_calculator_server(tool_prefix="calc")
    fs = create_filesystem_server(tool_prefix="fs")

    agent = Agent(
        model or get_model(),
        system_prompt=(
            "You have calculator tools (calc_add, calc_multiply, etc) "
            "and filesystem tools (fs_list_directory, fs_read_file, etc). "
            "Use them to help the user."
        ),
        toolsets=[calc, fs],
    )

    async with agent:
        result = await agent.run("What is 6 * 7? Also, does /tmp exist?")
        return result.output


# --8<-- [end:run]


if __name__ == "__main__":

    async def main() -> None:
        print("=" * 60)
        print("DEMO: MCP Integration Pattern")
        print("=" * 60)
        print()

        # Simple calculator example
        print("1. Simple calculator tool:")
        print("-" * 40)
        result = await run_with_mcp_tools("What is 42 * 17?")
        print(f"Result: {result}")
        print()

        # Multi-server example
        print("2. Multi-server (calculator + filesystem):")
        print("-" * 40)
        result = await run_multi_server_example()
        print(f"Result: {result}")
        print()

        print("=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)

    asyncio.run(main())
