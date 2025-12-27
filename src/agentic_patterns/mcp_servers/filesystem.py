"""
Simple filesystem MCP server.

Run standalone:
    python -m agentic_patterns.mcp_servers.filesystem

This server exposes basic file operations as MCP tools.
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

app = FastMCP("Filesystem")


@app.tool()
def list_directory(path: str) -> list[str]:
    """List files and directories in a path."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Path does not exist: {path}")
    if not p.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    return sorted(f.name for f in p.iterdir())


@app.tool()
def read_file(path: str, max_chars: int = 10000) -> str:
    """Read file contents (truncated to max_chars)."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"File does not exist: {path}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {path}")
    content = p.read_text()
    if len(content) > max_chars:
        truncated = content[:max_chars]
        return f"{truncated}\n... [truncated, {len(content)} total chars]"
    return content


@app.tool()
def file_exists(path: str) -> bool:
    """Check if a file or directory exists."""
    return Path(path).exists()


@app.tool()
def get_file_size(path: str) -> int:
    """Get file size in bytes."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Path does not exist: {path}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return p.stat().st_size


if __name__ == "__main__":
    app.run(transport="stdio")
