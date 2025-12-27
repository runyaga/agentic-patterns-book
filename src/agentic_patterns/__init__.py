"""
Agentic Design Patterns implemented with pydantic-ai.

A port of patterns from the "Agentic Design Patterns" book.
"""

from agentic_patterns._models import get_model
from agentic_patterns.exception_recovery import ErrorCategory
from agentic_patterns.exception_recovery import RecoveryConfig
from agentic_patterns.exception_recovery import recoverable_run
from agentic_patterns.mcp_integration import MCPDeps
from agentic_patterns.mcp_integration import create_mcp_agent
from agentic_patterns.mcp_integration import run_with_mcp_tools

__all__ = [
    "get_model",
    # Exception Recovery (Ch 12)
    "recoverable_run",
    "RecoveryConfig",
    "ErrorCategory",
    # MCP Integration (Ch 10)
    "run_with_mcp_tools",
    "create_mcp_agent",
    "MCPDeps",
]
__version__ = "0.1.0"
