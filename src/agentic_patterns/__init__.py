"""
Agentic Design Patterns implemented with pydantic-ai.

A port of patterns from the "Agentic Design Patterns" book.
"""

from agentic_patterns._models import get_model
from agentic_patterns.exception_recovery import ErrorCategory
from agentic_patterns.exception_recovery import RecoveryConfig
from agentic_patterns.exception_recovery import recoverable_run

__all__ = [
    "get_model",
    "recoverable_run",
    "RecoveryConfig",
    "ErrorCategory",
]
__version__ = "0.1.0"
