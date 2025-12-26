"""
Human-in-the-Loop Pattern Implementation.

Based on the Agentic Design Patterns book Chapter 13:
Integrate human intelligence and judgment into AI workflows.

Key concepts:
- Human Oversight: Monitor AI performance and output
- Intervention: Request human help for errors or ambiguous scenarios
- Decision Augmentation: AI provides analysis, human makes final decision
- Escalation Policies: Protocols for when to hand off to human

This module implements:
- EscalationPolicy: Rules for when to escalate to human
- ApprovalWorkflow: Request and capture human decisions
- HumanFeedbackLoop: Collect feedback for improvement

Example usage:
    policy = EscalationPolicy(confidence_threshold=0.7)
    workflow = ApprovalWorkflow(policy=policy)
    result = await workflow.process_with_review(task, agent_output)
"""

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent

from agentic_patterns._models import get_model


# --8<-- [start:models]
class EscalationReason(str, Enum):
    """Reason for escalating to human review."""

    LOW_CONFIDENCE = "low_confidence"
    SENSITIVE_CONTENT = "sensitive_content"
    HIGH_RISK_ACTION = "high_risk_action"
    AMBIGUOUS_INPUT = "ambiguous_input"
    ERROR_OCCURRED = "error_occurred"
    POLICY_REQUIRED = "policy_required"
    USER_REQUESTED = "user_requested"


class ReviewDecision(str, Enum):
    """Human reviewer's decision."""

    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    ESCALATE_FURTHER = "escalate_further"


class TaskStatus(str, Enum):
    """Status of a task in the workflow."""

    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    COMPLETED = "completed"


class AgentOutput(BaseModel):
    """Output from an AI agent with confidence score."""

    content: str = Field(description="The agent's response content")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)",
    )
    reasoning: str = Field(
        default="",
        description="Explanation of the reasoning",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class EscalationRequest(BaseModel):
    """Request for human review."""

    task_id: str = Field(description="Unique task identifier")
    task_description: str = Field(description="Description of the task")
    agent_output: AgentOutput = Field(description="Output from the agent")
    escalation_reason: EscalationReason = Field(
        description="Why escalation was triggered",
    )
    context: str = Field(default="", description="Additional context")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When escalation was requested",
    )


class HumanReview(BaseModel):
    """Human reviewer's response."""

    task_id: str = Field(description="Task being reviewed")
    decision: ReviewDecision = Field(description="Reviewer's decision")
    feedback: str = Field(default="", description="Feedback from reviewer")
    modified_content: str = Field(
        default="",
        description="Modified content if decision is MODIFY",
    )
    reviewer_id: str = Field(default="", description="ID of the reviewer")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When review was completed",
    )


class DecisionAugmentation(BaseModel):
    """AI-augmented decision support."""

    options: list[str] = Field(description="Available options")
    analysis: str = Field(description="AI analysis of the situation")
    recommendation: str = Field(description="AI's recommended option")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in recommendation",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Identified risks",
    )
    considerations: list[str] = Field(
        default_factory=list,
        description="Key considerations for decision",
    )


class WorkflowStats(BaseModel):
    """Statistics about the approval workflow."""

    total_tasks: int = Field(description="Total tasks processed")
    auto_approved: int = Field(description="Tasks auto-approved")
    escalated: int = Field(description="Tasks escalated for review")
    human_approved: int = Field(description="Tasks approved by human")
    human_rejected: int = Field(description="Tasks rejected by human")
    human_modified: int = Field(description="Tasks modified by human")
    avg_confidence: float = Field(description="Average confidence score")


@dataclass
class EscalationPolicy:
    """
    Policy defining when to escalate tasks to human review.

    Configures thresholds and rules for automatic vs manual processing.
    """

    confidence_threshold: float = 0.7
    sensitive_keywords: list[str] = field(default_factory=list)
    high_risk_actions: list[str] = field(default_factory=list)
    always_review_types: list[str] = field(default_factory=list)
    custom_rules: list[Callable[[AgentOutput], bool]] = field(
        default_factory=list
    )

    def should_escalate(
        self,
        output: AgentOutput,
        task_type: str = "",
    ) -> tuple[bool, EscalationReason | None]:
        """
        Determine if a task should be escalated.

        Args:
            output: The agent's output to evaluate.
            task_type: Type of task being processed.

        Returns:
            Tuple of (should_escalate, reason).
        """
        # Check confidence threshold
        if output.confidence < self.confidence_threshold:
            return True, EscalationReason.LOW_CONFIDENCE

        # Check for sensitive content
        content_lower = output.content.lower()
        for keyword in self.sensitive_keywords:
            if keyword.lower() in content_lower:
                return True, EscalationReason.SENSITIVE_CONTENT

        # Check for high-risk actions
        for action in self.high_risk_actions:
            if action.lower() in content_lower:
                return True, EscalationReason.HIGH_RISK_ACTION

        # Check if task type requires review
        if task_type in self.always_review_types:
            return True, EscalationReason.POLICY_REQUIRED

        # Apply custom rules
        for rule in self.custom_rules:
            if rule(output):
                return True, EscalationReason.POLICY_REQUIRED

        return False, None


@dataclass
class ApprovalWorkflow:
    """
    Workflow for processing tasks with human approval gates.

    Manages the flow from agent output to final approved result.
    """

    policy: EscalationPolicy = field(default_factory=EscalationPolicy)
    pending_reviews: list[EscalationRequest] = field(default_factory=list)
    completed_reviews: list[HumanReview] = field(default_factory=list)
    task_counter: int = 0

    def _generate_task_id(self) -> str:
        """Generate a unique task ID."""
        self.task_counter += 1
        return f"task_{self.task_counter:04d}"

    def submit_for_review(
        self,
        output: AgentOutput,
        task_description: str,
        reason: EscalationReason,
        context: str = "",
    ) -> EscalationRequest:
        """
        Submit a task for human review.

        Args:
            output: Agent output to review.
            task_description: Description of the task.
            reason: Why the task needs review.
            context: Additional context.

        Returns:
            The created EscalationRequest.
        """
        request = EscalationRequest(
            task_id=self._generate_task_id(),
            task_description=task_description,
            agent_output=output,
            escalation_reason=reason,
            context=context,
        )
        self.pending_reviews.append(request)
        return request

    def get_pending_reviews(self) -> list[EscalationRequest]:
        """Get all pending review requests."""
        return self.pending_reviews.copy()

    def process_review(
        self,
        task_id: str,
        decision: ReviewDecision,
        feedback: str = "",
        modified_content: str = "",
        reviewer_id: str = "",
    ) -> HumanReview | None:
        """
        Process a human review decision.

        Args:
            task_id: ID of the task being reviewed.
            decision: The reviewer's decision.
            feedback: Optional feedback.
            modified_content: Modified content if applicable.
            reviewer_id: ID of the reviewer.

        Returns:
            The HumanReview record, or None if task not found.
        """
        # Find the pending request
        request = None
        for req in self.pending_reviews:
            if req.task_id == task_id:
                request = req
                break

        if not request:
            return None

        # Create review record
        review = HumanReview(
            task_id=task_id,
            decision=decision,
            feedback=feedback,
            modified_content=modified_content,
            reviewer_id=reviewer_id,
        )

        # Move from pending to completed
        self.pending_reviews.remove(request)
        self.completed_reviews.append(review)

        return review

    def get_final_output(
        self,
        output: AgentOutput,
        review: HumanReview | None,
    ) -> str:
        """
        Get the final output after review.

        Args:
            output: Original agent output.
            review: Human review if any.

        Returns:
            Final approved content.
        """
        if review is None:
            return output.content

        if review.decision == ReviewDecision.APPROVE:
            return output.content
        elif review.decision == ReviewDecision.MODIFY:
            return review.modified_content or output.content
        elif review.decision == ReviewDecision.REJECT:
            return ""
        else:
            return output.content

    def get_stats(self) -> WorkflowStats:
        """Get workflow statistics."""
        total = self.task_counter
        escalated = len(self.completed_reviews) + len(self.pending_reviews)
        auto_approved = total - escalated

        approved = sum(
            1
            for r in self.completed_reviews
            if r.decision == ReviewDecision.APPROVE
        )
        rejected = sum(
            1
            for r in self.completed_reviews
            if r.decision == ReviewDecision.REJECT
        )
        modified = sum(
            1
            for r in self.completed_reviews
            if r.decision == ReviewDecision.MODIFY
        )

        return WorkflowStats(
            total_tasks=total,
            auto_approved=auto_approved,
            escalated=escalated,
            human_approved=approved,
            human_rejected=rejected,
            human_modified=modified,
            avg_confidence=0.0,  # Would need to track
        )


@dataclass
class HumanFeedbackLoop:
    """
    Feedback collection system for continuous improvement.

    Records human feedback to improve future agent responses.
    """

    feedback_records: list[dict] = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)

    def record_feedback(
        self,
        task_id: str,
        original_output: str,
        feedback: str,
        was_helpful: bool,
        category: str = "",
    ) -> dict:
        """
        Record feedback from human reviewer.

        Args:
            task_id: ID of the reviewed task.
            original_output: What the agent produced.
            feedback: Human feedback text.
            was_helpful: Whether the output was helpful.
            category: Category of feedback.

        Returns:
            The recorded feedback entry.
        """
        record = {
            "task_id": task_id,
            "original_output": original_output,
            "feedback": feedback,
            "was_helpful": was_helpful,
            "category": category,
            "timestamp": datetime.now(),
        }
        self.feedback_records.append(record)
        return record

    def add_improvement_suggestion(self, suggestion: str) -> None:
        """Add an improvement suggestion."""
        self.improvement_suggestions.append(suggestion)

    def get_feedback_summary(self) -> dict:
        """Get summary of collected feedback."""
        total = len(self.feedback_records)
        helpful = sum(1 for r in self.feedback_records if r["was_helpful"])

        categories: dict[str, int] = {}
        for r in self.feedback_records:
            cat = r.get("category", "uncategorized")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_feedback": total,
            "helpful_count": helpful,
            "helpful_rate": helpful / total if total > 0 else 0.0,
            "categories": categories,
            "improvement_suggestions": len(self.improvement_suggestions),
        }

    def get_improvement_areas(self) -> list[str]:
        """Get areas needing improvement based on feedback."""
        # Find categories with low helpfulness
        category_stats: dict[str, dict] = {}
        for r in self.feedback_records:
            cat = r.get("category", "uncategorized")
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "helpful": 0}
            category_stats[cat]["total"] += 1
            if r["was_helpful"]:
                category_stats[cat]["helpful"] += 1

        problem_areas = []
        for cat, stats in category_stats.items():
            rate = stats["helpful"] / stats["total"] if stats["total"] else 0
            if rate < 0.5 and stats["total"] >= 3:
                problem_areas.append(f"{cat}: {rate:.0%} helpful")

        return problem_areas


# --8<-- [end:models]


# --8<-- [start:agents]
# Initialize model
model = get_model()

# Confidence evaluator agent
confidence_evaluator = Agent(
    model,
    system_prompt=(
        "You are a confidence evaluator. Given an AI agent's response, "
        "assess how confident the agent should be in this response. "
        "Consider accuracy, completeness, and potential for errors. "
        "Rate confidence from 0.0 (very uncertain) to 1.0 (very certain)."
    ),
    output_type=AgentOutput,
)

# Decision augmentation agent
decision_agent = Agent(
    model,
    system_prompt=(
        "You are a decision support assistant. Given a situation with "
        "multiple options, provide analysis to help a human make a decision. "
        "List the options, analyze pros/cons, identify risks, and make a "
        "recommendation while acknowledging that the human has final say."
    ),
    output_type=DecisionAugmentation,
)

# Task execution agent - returns AgentOutput with self-assessed confidence
task_agent = Agent(
    model,
    system_prompt=(
        "You are a helpful assistant. Complete the given task to the best "
        "of your ability. Be clear about any uncertainties or limitations. "
        "Self-assess your confidence (0.0-1.0) based on: "
        "- How certain you are about the accuracy of your response "
        "- Whether you have complete information to answer "
        "- The complexity and ambiguity of the task "
        "A score of 0.9+ means very confident, 0.7-0.9 means reasonably "
        "confident, below 0.7 means uncertain."
    ),
    output_type=AgentOutput,
)
# --8<-- [end:agents]


# --8<-- [start:workflow]
async def execute_with_oversight(
    task: str,
    policy: EscalationPolicy,
    workflow: ApprovalWorkflow,
    task_type: str = "",
) -> tuple[str, bool, EscalationRequest | None]:
    """
    Execute a task with human oversight based on policy.

    Args:
        task: The task to execute.
        policy: Escalation policy to apply.
        workflow: Approval workflow to use.
        task_type: Type of task for policy checks.

    Returns:
        Tuple of (result, was_escalated, escalation_request).
    """
    print(f"Executing task: {task[:50]}...")

    # Execute the task - model self-assesses confidence
    result = await task_agent.run(task)
    output = result.output  # AgentOutput with self-assessed confidence

    # Check escalation policy
    should_escalate, reason = policy.should_escalate(output, task_type)

    if should_escalate and reason:
        print(f"  Escalating: {reason.value}")
        request = workflow.submit_for_review(
            output=output,
            task_description=task,
            reason=reason,
        )
        return output.content, True, request

    print("  Auto-approved")
    workflow.task_counter += 1
    return output.content, False, None


async def augment_decision(
    situation: str,
    options: list[str],
) -> DecisionAugmentation:
    """
    Provide AI-augmented decision support.

    Args:
        situation: Description of the situation.
        options: Available options to choose from.

    Returns:
        DecisionAugmentation with analysis and recommendation.
    """
    options_text = "\n".join(f"- {opt}" for opt in options)

    result = await decision_agent.run(
        f"Situation: {situation}\n\n"
        f"Available options:\n{options_text}\n\n"
        f"Provide analysis and a recommendation to help the human decide."
    )

    return result.output


async def process_with_feedback(
    task: str,
    feedback_loop: HumanFeedbackLoop,
    category: str = "",
    simulated_feedback: tuple[str, bool] | None = None,
) -> tuple[str, dict]:
    """
    Execute task and collect feedback for improvement.

    Args:
        task: The task to execute.
        feedback_loop: Feedback collection system.
        category: Category for the feedback.
        simulated_feedback: Optional (feedback_text, was_helpful) for demo.

    Returns:
        Tuple of (output, feedback_record).
    """
    result = await task_agent.run(task)
    output = result.output

    # In a real system, this would wait for human feedback
    # For demo purposes, we use simulated feedback
    if simulated_feedback:
        feedback_text, was_helpful = simulated_feedback
    else:
        feedback_text = "No feedback provided"
        was_helpful = True

    record = feedback_loop.record_feedback(
        task_id=f"fb_{len(feedback_loop.feedback_records)}",
        original_output=output,
        feedback=feedback_text,
        was_helpful=was_helpful,
        category=category,
    )

    return output, record


# --8<-- [end:workflow]


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        print("=" * 60)
        print("DEMO: Human-in-the-Loop Pattern")
        print("=" * 60)

        # Create policy with escalation rules
        policy = EscalationPolicy(
            confidence_threshold=0.7,
            sensitive_keywords=["delete", "financial", "personal"],
            high_risk_actions=["execute", "deploy", "publish"],
        )

        workflow = ApprovalWorkflow(policy=policy)

        # Demo 1: Task that passes automatically
        print("\n--- Demo 1: Auto-Approved Task ---")
        result, escalated, request = await execute_with_oversight(
            task="Summarize the benefits of exercise",
            policy=policy,
            workflow=workflow,
        )
        print(f"Result: {result[:100]}...")
        print(f"Escalated: {escalated}")

        # Demo 2: Task with sensitive content
        print("\n--- Demo 2: Sensitive Content Task ---")
        result, escalated, request = await execute_with_oversight(
            task="How to delete user personal data from the database",
            policy=policy,
            workflow=workflow,
        )
        print(f"Result: {result[:100]}...")
        print(f"Escalated: {escalated}")
        if request:
            print(f"Reason: {request.escalation_reason.value}")

        # Simulate human review
        if request:
            review = workflow.process_review(
                task_id=request.task_id,
                decision=ReviewDecision.APPROVE,
                feedback="Reviewed and approved for development use",
                reviewer_id="human_1",
            )
            print(f"Review decision: {review.decision.value}")

        # Demo 3: Decision augmentation
        print("\n--- Demo 3: Decision Augmentation ---")
        augmentation = await augment_decision(
            situation="Need to choose a database for a new web application",
            options=["PostgreSQL", "MongoDB", "SQLite"],
        )
        print(f"Recommendation: {augmentation.recommendation}")
        print(f"Confidence: {augmentation.confidence:.0%}")

        # Show workflow stats
        stats = workflow.get_stats()
        print("\n--- Workflow Statistics ---")
        print(f"Total tasks: {stats.total_tasks}")
        print(f"Auto-approved: {stats.auto_approved}")
        print(f"Escalated: {stats.escalated}")
        print(f"Human approved: {stats.human_approved}")

    asyncio.run(main())
