"""
Agentic Design Patterns implemented with pydantic-ai.

A port of patterns from the "Agentic Design Patterns" book.
"""

from agentic_patterns._models import get_model
from agentic_patterns.agent_marketplace import AgentBid
from agentic_patterns.agent_marketplace import AgentCapability
from agentic_patterns.agent_marketplace import AgentJudgmentStrategy
from agentic_patterns.agent_marketplace import AgoraCallbacks
from agentic_patterns.agent_marketplace import AgoraConfig
from agentic_patterns.agent_marketplace import BestSkillMatchStrategy
from agentic_patterns.agent_marketplace import CapacityAwareStrategy
from agentic_patterns.agent_marketplace import HighestConfidenceStrategy
from agentic_patterns.agent_marketplace import SelectionStrategy
from agentic_patterns.agent_marketplace import TaskResult
from agentic_patterns.agent_marketplace import TaskRFP
from agentic_patterns.agent_marketplace import WeightedScoreStrategy
from agentic_patterns.agent_marketplace import create_bidder_agent
from agentic_patterns.agent_marketplace import run_marketplace_task
from agentic_patterns.exception_recovery import ErrorCategory
from agentic_patterns.exception_recovery import RecoveryConfig
from agentic_patterns.exception_recovery import is_retryable
from agentic_patterns.exception_recovery import recoverable_run
from agentic_patterns.goal_monitoring import Goal
from agentic_patterns.goal_monitoring import GoalMonitor
from agentic_patterns.goal_monitoring import GoalStatus
from agentic_patterns.goal_monitoring import on_escalate
from agentic_patterns.goal_monitoring import run_goal_monitor
from agentic_patterns.mcp_integration import MCPDeps
from agentic_patterns.mcp_integration import create_mcp_agent
from agentic_patterns.mcp_integration import run_with_mcp_tools

__all__ = [
    "get_model",
    # Agent Marketplace (Ch 15)
    "run_marketplace_task",
    "create_bidder_agent",
    "TaskRFP",
    "TaskResult",
    "AgentBid",
    "AgentCapability",
    # Selection Strategies (Ch 15b)
    "SelectionStrategy",
    "HighestConfidenceStrategy",
    "BestSkillMatchStrategy",
    "WeightedScoreStrategy",
    "AgentJudgmentStrategy",
    "CapacityAwareStrategy",
    # Production Features (Ch 15c)
    "AgoraCallbacks",
    "AgoraConfig",
    # Exception Recovery (Ch 12)
    "recoverable_run",
    "RecoveryConfig",
    "ErrorCategory",
    "is_retryable",
    # MCP Integration (Ch 10)
    "run_with_mcp_tools",
    "create_mcp_agent",
    "MCPDeps",
    # Goal Monitoring (Ch 11)
    "Goal",
    "GoalStatus",
    "GoalMonitor",
    "run_goal_monitor",
    "on_escalate",
]
__version__ = "0.7.0"
