"""Tests for the Human-in-the-Loop pattern module."""

from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns.human_in_loop import AgentOutput
from agentic_patterns.human_in_loop import ApprovalWorkflow
from agentic_patterns.human_in_loop import DecisionAugmentation
from agentic_patterns.human_in_loop import EscalationPolicy
from agentic_patterns.human_in_loop import EscalationReason
from agentic_patterns.human_in_loop import EscalationRequest
from agentic_patterns.human_in_loop import HumanFeedbackLoop
from agentic_patterns.human_in_loop import HumanReview
from agentic_patterns.human_in_loop import ReviewDecision
from agentic_patterns.human_in_loop import TaskStatus
from agentic_patterns.human_in_loop import WorkflowStats
from agentic_patterns.human_in_loop import augment_decision
from agentic_patterns.human_in_loop import execute_with_oversight
from agentic_patterns.human_in_loop import process_with_feedback


class TestEscalationReason:
    """Tests for EscalationReason enum."""

    def test_escalation_reason_values(self) -> None:
        """Test all escalation reason values exist."""
        assert EscalationReason.LOW_CONFIDENCE == "low_confidence"
        assert EscalationReason.SENSITIVE_CONTENT == "sensitive_content"
        assert EscalationReason.HIGH_RISK_ACTION == "high_risk_action"
        assert EscalationReason.AMBIGUOUS_INPUT == "ambiguous_input"
        assert EscalationReason.ERROR_OCCURRED == "error_occurred"
        assert EscalationReason.POLICY_REQUIRED == "policy_required"
        assert EscalationReason.USER_REQUESTED == "user_requested"


class TestReviewDecision:
    """Tests for ReviewDecision enum."""

    def test_review_decision_values(self) -> None:
        """Test all review decision values exist."""
        assert ReviewDecision.APPROVE == "approve"
        assert ReviewDecision.REJECT == "reject"
        assert ReviewDecision.MODIFY == "modify"
        assert ReviewDecision.ESCALATE_FURTHER == "escalate_further"


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_task_status_values(self) -> None:
        """Test all task status values exist."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_REVIEW == "in_review"
        assert TaskStatus.APPROVED == "approved"
        assert TaskStatus.REJECTED == "rejected"
        assert TaskStatus.MODIFIED == "modified"
        assert TaskStatus.COMPLETED == "completed"


class TestAgentOutput:
    """Tests for AgentOutput model."""

    def test_agent_output_creation(self) -> None:
        """Test creating agent output with all fields."""
        output = AgentOutput(
            content="Test response",
            confidence=0.85,
            reasoning="Based on analysis",
            metadata={"source": "test"},
        )
        assert output.content == "Test response"
        assert output.confidence == 0.85
        assert output.reasoning == "Based on analysis"
        assert output.metadata == {"source": "test"}

    def test_agent_output_defaults(self) -> None:
        """Test agent output default values."""
        output = AgentOutput(content="Test", confidence=0.5)
        assert output.reasoning == ""
        assert output.metadata == {}

    def test_agent_output_confidence_bounds(self) -> None:
        """Test confidence value constraints."""
        # Valid at boundaries
        output_low = AgentOutput(content="Test", confidence=0.0)
        assert output_low.confidence == 0.0

        output_high = AgentOutput(content="Test", confidence=1.0)
        assert output_high.confidence == 1.0

    def test_agent_output_invalid_confidence(self) -> None:
        """Test invalid confidence values are rejected."""
        with pytest.raises(ValueError):
            AgentOutput(content="Test", confidence=1.5)

        with pytest.raises(ValueError):
            AgentOutput(content="Test", confidence=-0.1)


class TestEscalationRequest:
    """Tests for EscalationRequest model."""

    def test_escalation_request_creation(self) -> None:
        """Test creating escalation request."""
        output = AgentOutput(content="Response", confidence=0.5)
        request = EscalationRequest(
            task_id="task_001",
            task_description="Process data",
            agent_output=output,
            escalation_reason=EscalationReason.LOW_CONFIDENCE,
            context="Additional info",
        )
        assert request.task_id == "task_001"
        assert request.task_description == "Process data"
        assert request.escalation_reason == EscalationReason.LOW_CONFIDENCE
        assert request.context == "Additional info"
        assert isinstance(request.timestamp, datetime)

    def test_escalation_request_defaults(self) -> None:
        """Test escalation request default values."""
        output = AgentOutput(content="Response", confidence=0.5)
        request = EscalationRequest(
            task_id="task_002",
            task_description="Task",
            agent_output=output,
            escalation_reason=EscalationReason.ERROR_OCCURRED,
        )
        assert request.context == ""


class TestHumanReview:
    """Tests for HumanReview model."""

    def test_human_review_creation(self) -> None:
        """Test creating human review."""
        review = HumanReview(
            task_id="task_001",
            decision=ReviewDecision.APPROVE,
            feedback="Looks good",
            modified_content="",
            reviewer_id="reviewer_1",
        )
        assert review.task_id == "task_001"
        assert review.decision == ReviewDecision.APPROVE
        assert review.feedback == "Looks good"
        assert review.reviewer_id == "reviewer_1"
        assert isinstance(review.timestamp, datetime)

    def test_human_review_defaults(self) -> None:
        """Test human review default values."""
        review = HumanReview(
            task_id="task_002",
            decision=ReviewDecision.REJECT,
        )
        assert review.feedback == ""
        assert review.modified_content == ""
        assert review.reviewer_id == ""

    def test_human_review_with_modification(self) -> None:
        """Test human review with modified content."""
        review = HumanReview(
            task_id="task_003",
            decision=ReviewDecision.MODIFY,
            modified_content="Updated response here",
        )
        assert review.decision == ReviewDecision.MODIFY
        assert review.modified_content == "Updated response here"


class TestDecisionAugmentation:
    """Tests for DecisionAugmentation model."""

    def test_decision_augmentation_creation(self) -> None:
        """Test creating decision augmentation."""
        aug = DecisionAugmentation(
            options=["Option A", "Option B"],
            analysis="Detailed analysis here",
            recommendation="Option A",
            confidence=0.75,
            risks=["Risk 1", "Risk 2"],
            considerations=["Consider X", "Consider Y"],
        )
        assert aug.options == ["Option A", "Option B"]
        assert aug.analysis == "Detailed analysis here"
        assert aug.recommendation == "Option A"
        assert aug.confidence == 0.75
        assert len(aug.risks) == 2
        assert len(aug.considerations) == 2

    def test_decision_augmentation_defaults(self) -> None:
        """Test decision augmentation default values."""
        aug = DecisionAugmentation(
            options=["A"],
            analysis="Analysis",
            recommendation="A",
            confidence=0.5,
        )
        assert aug.risks == []
        assert aug.considerations == []


class TestWorkflowStats:
    """Tests for WorkflowStats model."""

    def test_workflow_stats_creation(self) -> None:
        """Test creating workflow stats."""
        stats = WorkflowStats(
            total_tasks=100,
            auto_approved=80,
            escalated=20,
            human_approved=15,
            human_rejected=3,
            human_modified=2,
            avg_confidence=0.85,
        )
        assert stats.total_tasks == 100
        assert stats.auto_approved == 80
        assert stats.escalated == 20
        assert stats.human_approved == 15
        assert stats.human_rejected == 3
        assert stats.human_modified == 2
        assert stats.avg_confidence == 0.85


class TestEscalationPolicy:
    """Tests for EscalationPolicy dataclass."""

    def test_default_policy(self) -> None:
        """Test default policy values."""
        policy = EscalationPolicy()
        assert policy.confidence_threshold == 0.7
        assert policy.sensitive_keywords == []
        assert policy.high_risk_actions == []
        assert policy.always_review_types == []
        assert policy.custom_rules == []

    def test_custom_policy(self) -> None:
        """Test custom policy configuration."""
        policy = EscalationPolicy(
            confidence_threshold=0.8,
            sensitive_keywords=["password", "secret"],
            high_risk_actions=["delete", "deploy"],
            always_review_types=["financial"],
        )
        assert policy.confidence_threshold == 0.8
        assert "password" in policy.sensitive_keywords
        assert "delete" in policy.high_risk_actions
        assert "financial" in policy.always_review_types

    def test_should_escalate_low_confidence(self) -> None:
        """Test escalation triggered by low confidence."""
        policy = EscalationPolicy(confidence_threshold=0.7)
        output = AgentOutput(content="Response", confidence=0.5)

        should, reason = policy.should_escalate(output)
        assert should is True
        assert reason == EscalationReason.LOW_CONFIDENCE

    def test_should_not_escalate_high_confidence(self) -> None:
        """Test no escalation for high confidence."""
        policy = EscalationPolicy(confidence_threshold=0.7)
        output = AgentOutput(content="Response", confidence=0.9)

        should, reason = policy.should_escalate(output)
        assert should is False
        assert reason is None

    def test_should_escalate_sensitive_content(self) -> None:
        """Test escalation triggered by sensitive keywords."""
        policy = EscalationPolicy(
            confidence_threshold=0.5,
            sensitive_keywords=["password", "secret"],
        )
        output = AgentOutput(
            content="Here is your password reset",
            confidence=0.9,
        )

        should, reason = policy.should_escalate(output)
        assert should is True
        assert reason == EscalationReason.SENSITIVE_CONTENT

    def test_should_escalate_high_risk_action(self) -> None:
        """Test escalation triggered by high-risk action."""
        policy = EscalationPolicy(
            confidence_threshold=0.5,
            high_risk_actions=["delete", "execute"],
        )
        output = AgentOutput(
            content="Ready to delete the records",
            confidence=0.9,
        )

        should, reason = policy.should_escalate(output)
        assert should is True
        assert reason == EscalationReason.HIGH_RISK_ACTION

    def test_should_escalate_always_review_type(self) -> None:
        """Test escalation triggered by task type."""
        policy = EscalationPolicy(
            confidence_threshold=0.5,
            always_review_types=["financial", "legal"],
        )
        output = AgentOutput(content="Response", confidence=0.9)

        should, reason = policy.should_escalate(output, task_type="financial")
        assert should is True
        assert reason == EscalationReason.POLICY_REQUIRED

    def test_should_escalate_custom_rule(self) -> None:
        """Test escalation triggered by custom rule."""

        def long_response_rule(output: AgentOutput) -> bool:
            return len(output.content) > 100

        policy = EscalationPolicy(
            confidence_threshold=0.5,
            custom_rules=[long_response_rule],
        )
        output = AgentOutput(content="x" * 150, confidence=0.9)

        should, reason = policy.should_escalate(output)
        assert should is True
        assert reason == EscalationReason.POLICY_REQUIRED

    def test_case_insensitive_keyword_matching(self) -> None:
        """Test keyword matching is case insensitive."""
        policy = EscalationPolicy(
            confidence_threshold=0.5,
            sensitive_keywords=["PASSWORD"],
        )
        output = AgentOutput(
            content="Your password is ready",
            confidence=0.9,
        )

        should, reason = policy.should_escalate(output)
        assert should is True


class TestApprovalWorkflow:
    """Tests for ApprovalWorkflow dataclass."""

    def test_default_workflow(self) -> None:
        """Test default workflow initialization."""
        workflow = ApprovalWorkflow()
        assert isinstance(workflow.policy, EscalationPolicy)
        assert workflow.pending_reviews == []
        assert workflow.completed_reviews == []
        assert workflow.task_counter == 0

    def test_generate_task_id(self) -> None:
        """Test unique task ID generation."""
        workflow = ApprovalWorkflow()
        id1 = workflow._generate_task_id()
        id2 = workflow._generate_task_id()
        assert id1 == "task_0001"
        assert id2 == "task_0002"
        assert id1 != id2

    def test_submit_for_review(self) -> None:
        """Test submitting task for review."""
        workflow = ApprovalWorkflow()
        output = AgentOutput(content="Response", confidence=0.5)

        request = workflow.submit_for_review(
            output=output,
            task_description="Test task",
            reason=EscalationReason.LOW_CONFIDENCE,
            context="Extra context",
        )

        assert request.task_id == "task_0001"
        assert request.task_description == "Test task"
        assert request.escalation_reason == EscalationReason.LOW_CONFIDENCE
        assert request.context == "Extra context"
        assert len(workflow.pending_reviews) == 1

    def test_get_pending_reviews(self) -> None:
        """Test getting pending reviews."""
        workflow = ApprovalWorkflow()
        output = AgentOutput(content="Response", confidence=0.5)

        workflow.submit_for_review(
            output=output,
            task_description="Task 1",
            reason=EscalationReason.LOW_CONFIDENCE,
        )
        workflow.submit_for_review(
            output=output,
            task_description="Task 2",
            reason=EscalationReason.SENSITIVE_CONTENT,
        )

        pending = workflow.get_pending_reviews()
        assert len(pending) == 2
        # Verify it's a copy
        pending.pop()
        assert len(workflow.pending_reviews) == 2

    def test_process_review_approve(self) -> None:
        """Test processing an approval review."""
        workflow = ApprovalWorkflow()
        output = AgentOutput(content="Response", confidence=0.5)

        request = workflow.submit_for_review(
            output=output,
            task_description="Task",
            reason=EscalationReason.LOW_CONFIDENCE,
        )

        review = workflow.process_review(
            task_id=request.task_id,
            decision=ReviewDecision.APPROVE,
            feedback="Looks good",
            reviewer_id="reviewer_1",
        )

        assert review is not None
        assert review.decision == ReviewDecision.APPROVE
        assert review.feedback == "Looks good"
        assert len(workflow.pending_reviews) == 0
        assert len(workflow.completed_reviews) == 1

    def test_process_review_reject(self) -> None:
        """Test processing a rejection review."""
        workflow = ApprovalWorkflow()
        output = AgentOutput(content="Response", confidence=0.5)

        request = workflow.submit_for_review(
            output=output,
            task_description="Task",
            reason=EscalationReason.HIGH_RISK_ACTION,
        )

        review = workflow.process_review(
            task_id=request.task_id,
            decision=ReviewDecision.REJECT,
            feedback="Not appropriate",
        )

        assert review is not None
        assert review.decision == ReviewDecision.REJECT

    def test_process_review_modify(self) -> None:
        """Test processing a modification review."""
        workflow = ApprovalWorkflow()
        output = AgentOutput(content="Original response", confidence=0.6)

        request = workflow.submit_for_review(
            output=output,
            task_description="Task",
            reason=EscalationReason.SENSITIVE_CONTENT,
        )

        review = workflow.process_review(
            task_id=request.task_id,
            decision=ReviewDecision.MODIFY,
            modified_content="Modified response",
        )

        assert review is not None
        assert review.decision == ReviewDecision.MODIFY
        assert review.modified_content == "Modified response"

    def test_process_review_not_found(self) -> None:
        """Test processing review for non-existent task."""
        workflow = ApprovalWorkflow()
        review = workflow.process_review(
            task_id="nonexistent",
            decision=ReviewDecision.APPROVE,
        )
        assert review is None

    def test_get_final_output_no_review(self) -> None:
        """Test getting final output without review."""
        workflow = ApprovalWorkflow()
        output = AgentOutput(content="Original", confidence=0.9)

        result = workflow.get_final_output(output, None)
        assert result == "Original"

    def test_get_final_output_approved(self) -> None:
        """Test getting final output after approval."""
        workflow = ApprovalWorkflow()
        output = AgentOutput(content="Original", confidence=0.9)
        review = HumanReview(
            task_id="task_001",
            decision=ReviewDecision.APPROVE,
        )

        result = workflow.get_final_output(output, review)
        assert result == "Original"

    def test_get_final_output_modified(self) -> None:
        """Test getting final output after modification."""
        workflow = ApprovalWorkflow()
        output = AgentOutput(content="Original", confidence=0.9)
        review = HumanReview(
            task_id="task_001",
            decision=ReviewDecision.MODIFY,
            modified_content="Modified content",
        )

        result = workflow.get_final_output(output, review)
        assert result == "Modified content"

    def test_get_final_output_rejected(self) -> None:
        """Test getting final output after rejection."""
        workflow = ApprovalWorkflow()
        output = AgentOutput(content="Original", confidence=0.9)
        review = HumanReview(
            task_id="task_001",
            decision=ReviewDecision.REJECT,
        )

        result = workflow.get_final_output(output, review)
        assert result == ""

    def test_get_stats(self) -> None:
        """Test getting workflow statistics."""
        workflow = ApprovalWorkflow()
        output = AgentOutput(content="Response", confidence=0.5)

        # Add some tasks
        req1 = workflow.submit_for_review(
            output=output,
            task_description="Task 1",
            reason=EscalationReason.LOW_CONFIDENCE,
        )
        req2 = workflow.submit_for_review(
            output=output,
            task_description="Task 2",
            reason=EscalationReason.SENSITIVE_CONTENT,
        )
        req3 = workflow.submit_for_review(
            output=output,
            task_description="Task 3",
            reason=EscalationReason.HIGH_RISK_ACTION,
        )

        # Process some reviews
        workflow.process_review(req1.task_id, ReviewDecision.APPROVE)
        workflow.process_review(req2.task_id, ReviewDecision.REJECT)
        workflow.process_review(req3.task_id, ReviewDecision.MODIFY)

        # Add some auto-approved tasks
        workflow.task_counter += 2

        stats = workflow.get_stats()
        assert stats.total_tasks == 5
        assert stats.escalated == 3
        assert stats.auto_approved == 2
        assert stats.human_approved == 1
        assert stats.human_rejected == 1
        assert stats.human_modified == 1


class TestHumanFeedbackLoop:
    """Tests for HumanFeedbackLoop dataclass."""

    def test_default_feedback_loop(self) -> None:
        """Test default feedback loop initialization."""
        loop = HumanFeedbackLoop()
        assert loop.feedback_records == []
        assert loop.improvement_suggestions == []

    def test_record_feedback(self) -> None:
        """Test recording feedback."""
        loop = HumanFeedbackLoop()
        record = loop.record_feedback(
            task_id="task_001",
            original_output="AI response",
            feedback="Could be clearer",
            was_helpful=True,
            category="clarity",
        )

        assert record["task_id"] == "task_001"
        assert record["original_output"] == "AI response"
        assert record["feedback"] == "Could be clearer"
        assert record["was_helpful"] is True
        assert record["category"] == "clarity"
        assert "timestamp" in record
        assert len(loop.feedback_records) == 1

    def test_add_improvement_suggestion(self) -> None:
        """Test adding improvement suggestions."""
        loop = HumanFeedbackLoop()
        loop.add_improvement_suggestion("Add more examples")
        loop.add_improvement_suggestion("Reduce verbosity")

        assert len(loop.improvement_suggestions) == 2
        assert "Add more examples" in loop.improvement_suggestions

    def test_get_feedback_summary(self) -> None:
        """Test getting feedback summary."""
        loop = HumanFeedbackLoop()

        # Add mixed feedback
        loop.record_feedback("t1", "out1", "good", True, "quality")
        loop.record_feedback("t2", "out2", "bad", False, "quality")
        loop.record_feedback("t3", "out3", "ok", True, "clarity")

        loop.add_improvement_suggestion("Suggestion 1")

        summary = loop.get_feedback_summary()
        assert summary["total_feedback"] == 3
        assert summary["helpful_count"] == 2
        assert summary["helpful_rate"] == pytest.approx(2 / 3)
        assert summary["categories"]["quality"] == 2
        assert summary["categories"]["clarity"] == 1
        assert summary["improvement_suggestions"] == 1

    def test_get_feedback_summary_empty(self) -> None:
        """Test feedback summary with no feedback."""
        loop = HumanFeedbackLoop()
        summary = loop.get_feedback_summary()
        assert summary["total_feedback"] == 0
        assert summary["helpful_rate"] == 0.0

    def test_get_improvement_areas(self) -> None:
        """Test identifying areas needing improvement."""
        loop = HumanFeedbackLoop()

        # Category with low helpfulness
        loop.record_feedback("t1", "o1", "f1", False, "formatting")
        loop.record_feedback("t2", "o2", "f2", False, "formatting")
        loop.record_feedback("t3", "o3", "f3", False, "formatting")

        # Category with high helpfulness
        loop.record_feedback("t4", "o4", "f4", True, "accuracy")
        loop.record_feedback("t5", "o5", "f5", True, "accuracy")
        loop.record_feedback("t6", "o6", "f6", True, "accuracy")

        areas = loop.get_improvement_areas()
        assert any("formatting" in area for area in areas)
        assert not any("accuracy" in area for area in areas)

    def test_get_improvement_areas_minimum_data(self) -> None:
        """Test improvement areas requires minimum data points."""
        loop = HumanFeedbackLoop()

        # Only 2 data points - not enough
        loop.record_feedback("t1", "o1", "f1", False, "test")
        loop.record_feedback("t2", "o2", "f2", False, "test")

        areas = loop.get_improvement_areas()
        assert len(areas) == 0  # Not enough data


class TestExecuteWithOversight:
    """Tests for execute_with_oversight function."""

    @pytest.mark.asyncio
    async def test_execute_auto_approved(self) -> None:
        """Test task execution that passes without escalation."""
        policy = EscalationPolicy(confidence_threshold=0.7)
        workflow = ApprovalWorkflow(policy=policy)

        # Model now returns AgentOutput with self-assessed confidence
        mock_agent_output = AgentOutput(
            content="Task completed successfully",
            confidence=0.9,
            reasoning="High confidence response.",
        )
        mock_result = MagicMock()
        mock_result.output = mock_agent_output

        with patch("agentic_patterns.human_in_loop.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            result, escalated, request = await execute_with_oversight(
                task="Simple task",
                policy=policy,
                workflow=workflow,
            )

        assert result == "Task completed successfully"
        assert escalated is False
        assert request is None
        assert workflow.task_counter == 1

    @pytest.mark.asyncio
    async def test_execute_escalated_low_confidence(self) -> None:
        """Test task execution escalated due to low confidence."""
        policy = EscalationPolicy(confidence_threshold=0.8)
        workflow = ApprovalWorkflow(policy=policy)

        # Model self-assesses low confidence
        mock_agent_output = AgentOutput(
            content="Uncertain response",
            confidence=0.5,
            reasoning="I'm not sure about this.",
        )
        mock_result = MagicMock()
        mock_result.output = mock_agent_output

        with patch("agentic_patterns.human_in_loop.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            result, escalated, request = await execute_with_oversight(
                task="Complex task",
                policy=policy,
                workflow=workflow,
            )

        assert result == "Uncertain response"
        assert escalated is True
        assert request is not None
        assert request.escalation_reason == EscalationReason.LOW_CONFIDENCE

    @pytest.mark.asyncio
    async def test_execute_escalated_sensitive_content(self) -> None:
        """Test task execution escalated due to sensitive content."""
        policy = EscalationPolicy(
            confidence_threshold=0.5,
            sensitive_keywords=["delete", "password"],
        )
        workflow = ApprovalWorkflow(policy=policy)

        # Model returns high confidence but sensitive content
        mock_agent_output = AgentOutput(
            content="Here is how to delete your account",
            confidence=0.9,
            reasoning="Clear instructions provided.",
        )
        mock_result = MagicMock()
        mock_result.output = mock_agent_output

        with patch("agentic_patterns.human_in_loop.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            result, escalated, request = await execute_with_oversight(
                task="Account question",
                policy=policy,
                workflow=workflow,
            )

        assert escalated is True
        assert request is not None
        assert request.escalation_reason == EscalationReason.SENSITIVE_CONTENT


class TestAugmentDecision:
    """Tests for augment_decision function."""

    @pytest.mark.asyncio
    async def test_augment_decision(self) -> None:
        """Test decision augmentation."""
        mock_augmentation = DecisionAugmentation(
            options=["PostgreSQL", "MongoDB"],
            analysis="PostgreSQL is better for relational data",
            recommendation="PostgreSQL",
            confidence=0.8,
            risks=["Learning curve"],
            considerations=["Data structure"],
        )

        mock_result = MagicMock()
        mock_result.output = mock_augmentation

        with patch(
            "agentic_patterns.human_in_loop.decision_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            result = await augment_decision(
                situation="Choosing a database",
                options=["PostgreSQL", "MongoDB"],
            )

        assert result.recommendation == "PostgreSQL"
        assert result.confidence == 0.8
        assert "PostgreSQL" in result.options


class TestProcessWithFeedback:
    """Tests for process_with_feedback function."""

    @pytest.mark.asyncio
    async def test_process_with_simulated_feedback(self) -> None:
        """Test processing task with simulated feedback."""
        feedback_loop = HumanFeedbackLoop()

        mock_result = MagicMock()
        mock_result.output = "Generated response"

        with patch("agentic_patterns.human_in_loop.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            output, record = await process_with_feedback(
                task="Generate text",
                feedback_loop=feedback_loop,
                category="generation",
                simulated_feedback=("Great response!", True),
            )

        assert output == "Generated response"
        assert record["feedback"] == "Great response!"
        assert record["was_helpful"] is True
        assert record["category"] == "generation"
        assert len(feedback_loop.feedback_records) == 1

    @pytest.mark.asyncio
    async def test_process_without_simulated_feedback(self) -> None:
        """Test processing task without simulated feedback."""
        feedback_loop = HumanFeedbackLoop()

        mock_result = MagicMock()
        mock_result.output = "Generated response"

        with patch("agentic_patterns.human_in_loop.task_agent") as mock_agent:
            mock_agent.run = AsyncMock(return_value=mock_result)

            output, record = await process_with_feedback(
                task="Generate text",
                feedback_loop=feedback_loop,
            )

        assert output == "Generated response"
        assert record["feedback"] == "No feedback provided"
        assert record["was_helpful"] is True  # Default


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_approval_workflow(self) -> None:
        """Test complete workflow from submission to final output."""
        policy = EscalationPolicy(
            confidence_threshold=0.8,
            sensitive_keywords=["confidential"],
        )
        workflow = ApprovalWorkflow(policy=policy)

        # Submit task for review
        output = AgentOutput(
            content="This is confidential information",
            confidence=0.9,
        )

        should_escalate, reason = policy.should_escalate(output)
        assert should_escalate is True

        request = workflow.submit_for_review(
            output=output,
            task_description="Share confidential data",
            reason=reason,
        )

        # Human reviews and modifies
        review = workflow.process_review(
            task_id=request.task_id,
            decision=ReviewDecision.MODIFY,
            modified_content="This is [REDACTED] information",
            feedback="Removed sensitive details",
            reviewer_id="security_team",
        )

        # Get final output
        final = workflow.get_final_output(output, review)
        assert final == "This is [REDACTED] information"

        # Check stats
        stats = workflow.get_stats()
        assert stats.human_modified == 1

    def test_feedback_loop_improvement_cycle(self) -> None:
        """Test feedback loop identifying improvement areas."""
        loop = HumanFeedbackLoop()

        # Simulate repeated poor performance in one category
        for i in range(5):
            loop.record_feedback(
                task_id=f"task_{i}",
                original_output=f"Response {i}",
                feedback="Too verbose",
                was_helpful=False,
                category="conciseness",
            )

        # Good performance in another category
        for i in range(5):
            loop.record_feedback(
                task_id=f"task_good_{i}",
                original_output=f"Good response {i}",
                feedback="Perfect",
                was_helpful=True,
                category="accuracy",
            )

        # Add improvement suggestion
        loop.add_improvement_suggestion("Be more concise in responses")

        # Check results
        areas = loop.get_improvement_areas()
        assert any("conciseness" in area for area in areas)

        summary = loop.get_feedback_summary()
        assert summary["helpful_rate"] == 0.5
