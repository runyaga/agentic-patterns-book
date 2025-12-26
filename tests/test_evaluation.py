"""Tests for the Evaluation and Monitoring module."""

from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from agentic_patterns.evaluation import ABTestResult
from agentic_patterns.evaluation import ABTestRunner
from agentic_patterns.evaluation import AgentMetrics
from agentic_patterns.evaluation import DriftDetector
from agentic_patterns.evaluation import DriftReport
from agentic_patterns.evaluation import DriftSeverity
from agentic_patterns.evaluation import EvaluationReport
from agentic_patterns.evaluation import EvaluationResult
from agentic_patterns.evaluation import JudgmentResult
from agentic_patterns.evaluation import LLMJudge
from agentic_patterns.evaluation import MetricType
from agentic_patterns.evaluation import MetricValue
from agentic_patterns.evaluation import PerformanceMonitor
from agentic_patterns.evaluation import PerformanceSummary
from agentic_patterns.evaluation import TrajectoryEvaluator
from agentic_patterns.evaluation import TrajectoryMatchType
from agentic_patterns.evaluation import TrajectoryResult
from agentic_patterns.evaluation import TrajectoryStep
from agentic_patterns.evaluation import calculate_precision_recall
from agentic_patterns.evaluation import evaluate_response_accuracy
from agentic_patterns.evaluation import generate_evaluation_report


class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_type_values(self) -> None:
        """Test all metric type values exist."""
        assert MetricType.ACCURACY == "accuracy"
        assert MetricType.LATENCY == "latency"
        assert MetricType.TOKEN_USAGE == "token_usage"
        assert MetricType.SUCCESS_RATE == "success_rate"
        assert MetricType.ERROR_RATE == "error_rate"
        assert MetricType.THROUGHPUT == "throughput"


class TestTrajectoryMatchType:
    """Tests for TrajectoryMatchType enum."""

    def test_match_type_values(self) -> None:
        """Test all match type values exist."""
        assert TrajectoryMatchType.EXACT == "exact"
        assert TrajectoryMatchType.IN_ORDER == "in_order"
        assert TrajectoryMatchType.ANY_ORDER == "any_order"
        assert TrajectoryMatchType.SINGLE_TOOL == "single_tool"


class TestDriftSeverity:
    """Tests for DriftSeverity enum."""

    def test_severity_values(self) -> None:
        """Test all severity values exist."""
        assert DriftSeverity.NONE == "none"
        assert DriftSeverity.LOW == "low"
        assert DriftSeverity.MEDIUM == "medium"
        assert DriftSeverity.HIGH == "high"
        assert DriftSeverity.CRITICAL == "critical"


class TestMetricValue:
    """Tests for MetricValue model."""

    def test_metric_value_creation(self) -> None:
        """Test creating a metric value."""
        metric = MetricValue(
            metric_type=MetricType.ACCURACY,
            value=0.95,
        )
        assert metric.metric_type == MetricType.ACCURACY
        assert metric.value == 0.95
        assert metric.metadata == {}
        assert isinstance(metric.timestamp, datetime)

    def test_metric_value_with_metadata(self) -> None:
        """Test metric value with metadata."""
        metric = MetricValue(
            metric_type=MetricType.LATENCY,
            value=150.5,
            metadata={"endpoint": "/api/query"},
        )
        assert metric.metadata["endpoint"] == "/api/query"


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_passing_result(self) -> None:
        """Test creating a passing evaluation result."""
        result = EvaluationResult(
            score=0.95,
            passed=True,
            details="High accuracy match",
        )
        assert result.score == 0.95
        assert result.passed is True
        assert result.metric_type == MetricType.ACCURACY

    def test_failing_result(self) -> None:
        """Test creating a failing evaluation result."""
        result = EvaluationResult(
            score=0.3,
            passed=False,
            details="Response differs significantly",
        )
        assert result.score == 0.3
        assert result.passed is False


class TestTrajectoryStep:
    """Tests for TrajectoryStep model."""

    def test_step_creation(self) -> None:
        """Test creating a trajectory step."""
        step = TrajectoryStep(
            action="search_database",
            tool_name="database",
            input_data="query: users",
        )
        assert step.action == "search_database"
        assert step.tool_name == "database"
        assert step.input_data == "query: users"
        assert step.duration_ms == 0.0

    def test_step_minimal(self) -> None:
        """Test trajectory step with minimal data."""
        step = TrajectoryStep(action="process")
        assert step.action == "process"
        assert step.tool_name is None


class TestTrajectoryResult:
    """Tests for TrajectoryResult model."""

    def test_result_creation(self) -> None:
        """Test creating a trajectory result."""
        result = TrajectoryResult(
            match_type=TrajectoryMatchType.IN_ORDER,
            match_score=0.75,
            matched_steps=["step1", "step2"],
            missing_steps=["step3"],
            extra_steps=["extra1"],
            details="Partial match",
        )
        assert result.match_type == TrajectoryMatchType.IN_ORDER
        assert result.match_score == 0.75
        assert len(result.matched_steps) == 2
        assert len(result.missing_steps) == 1


class TestJudgmentResult:
    """Tests for JudgmentResult model."""

    def test_judgment_creation(self) -> None:
        """Test creating a judgment result."""
        judgment = JudgmentResult(
            quality_score=0.9,
            helpfulness_score=0.85,
            accuracy_score=0.95,
            reasoning="Well-formed response with accurate info",
            improvement_suggestions=["Add more examples"],
        )
        assert judgment.quality_score == 0.9
        assert judgment.helpfulness_score == 0.85
        assert len(judgment.improvement_suggestions) == 1


class TestDriftReport:
    """Tests for DriftReport model."""

    def test_drift_report_creation(self) -> None:
        """Test creating a drift report."""
        report = DriftReport(
            severity=DriftSeverity.MEDIUM,
            affected_metrics=[MetricType.ACCURACY, MetricType.LATENCY],
            baseline_values={"accuracy": 0.95, "latency": 100.0},
            current_values={"accuracy": 0.80, "latency": 150.0},
            drift_percentage=15.7,
            recommendation="Investigate accuracy drop",
        )
        assert report.severity == DriftSeverity.MEDIUM
        assert len(report.affected_metrics) == 2
        assert report.drift_percentage == 15.7


class TestABTestResult:
    """Tests for ABTestResult model."""

    def test_ab_result_creation(self) -> None:
        """Test creating an A/B test result."""
        result = ABTestResult(
            variant_a_name="Original",
            variant_b_name="New",
            winner="New",
            variant_a_score=0.85,
            variant_b_score=0.92,
            confidence=0.95,
            sample_size=100,
            details="New variant shows improvement",
        )
        assert result.winner == "New"
        assert result.confidence == 0.95

    def test_ab_result_no_winner(self) -> None:
        """Test A/B result with no clear winner."""
        result = ABTestResult(
            variant_a_name="A",
            variant_b_name="B",
            winner=None,
            variant_a_score=0.86,
            variant_b_score=0.85,
            confidence=0.5,
            sample_size=20,
            details="No significant difference",
        )
        assert result.winner is None


class TestPerformanceSummary:
    """Tests for PerformanceSummary model."""

    def test_summary_creation(self) -> None:
        """Test creating a performance summary."""
        now = datetime.now()
        summary = PerformanceSummary(
            agent_id="agent-1",
            total_executions=100,
            avg_latency_ms=150.5,
            avg_accuracy=0.92,
            success_rate=0.95,
            total_tokens=15000,
            period_start=now,
            period_end=now,
        )
        assert summary.agent_id == "agent-1"
        assert summary.total_executions == 100
        assert summary.avg_latency_ms == 150.5


class TestEvaluationReport:
    """Tests for EvaluationReport model."""

    def test_report_creation(self) -> None:
        """Test creating an evaluation report."""
        report = EvaluationReport(
            agent_id="agent-1",
            recommendations=["Review low-scoring trajectories"],
        )
        assert report.agent_id == "agent-1"
        assert len(report.recommendations) == 1
        assert report.performance_summary is None
        assert report.drift_report is None


class TestAgentMetrics:
    """Tests for AgentMetrics dataclass."""

    def test_metrics_creation(self) -> None:
        """Test creating agent metrics."""
        metrics = AgentMetrics(agent_id="agent-1")
        assert metrics.agent_id == "agent-1"
        assert len(metrics.metrics) == 0

    def test_record_metric(self) -> None:
        """Test recording a metric."""
        metrics = AgentMetrics(agent_id="agent-1")
        metrics.record(MetricType.ACCURACY, 0.95)
        assert len(metrics.metrics) == 1
        assert metrics.metrics[0].value == 0.95

    def test_record_with_metadata(self) -> None:
        """Test recording metric with metadata."""
        metrics = AgentMetrics(agent_id="agent-1")
        metrics.record(MetricType.LATENCY, 100.0, {"request_id": "123"})
        assert metrics.metrics[0].metadata["request_id"] == "123"

    def test_get_average(self) -> None:
        """Test calculating metric average."""
        metrics = AgentMetrics(agent_id="agent-1")
        metrics.record(MetricType.ACCURACY, 0.90)
        metrics.record(MetricType.ACCURACY, 0.95)
        metrics.record(MetricType.ACCURACY, 1.00)

        avg = metrics.get_average(MetricType.ACCURACY)
        assert avg == pytest.approx(0.95, rel=0.01)

    def test_get_average_no_data(self) -> None:
        """Test average with no data returns None."""
        metrics = AgentMetrics(agent_id="agent-1")
        assert metrics.get_average(MetricType.ACCURACY) is None

    def test_get_recent(self) -> None:
        """Test getting recent metric values."""
        metrics = AgentMetrics(agent_id="agent-1")
        for i in range(15):
            metrics.record(MetricType.LATENCY, float(i * 10))

        recent = metrics.get_recent(MetricType.LATENCY, 5)
        assert len(recent) == 5
        assert recent == [100.0, 110.0, 120.0, 130.0, 140.0]

    def test_get_summary(self) -> None:
        """Test generating performance summary."""
        metrics = AgentMetrics(agent_id="agent-1")
        metrics.record(MetricType.LATENCY, 100.0)
        metrics.record(MetricType.LATENCY, 200.0)
        metrics.record(MetricType.ACCURACY, 0.9)
        metrics.record(MetricType.TOKEN_USAGE, 500.0)

        summary = metrics.get_summary()
        assert summary.agent_id == "agent-1"
        assert summary.avg_latency_ms == 150.0
        assert summary.avg_accuracy == 0.9
        assert summary.total_tokens == 500


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor dataclass."""

    def test_monitor_creation(self) -> None:
        """Test creating a performance monitor."""
        monitor = PerformanceMonitor()
        assert len(monitor.agents) == 0
        assert len(monitor.alerts) == 0

    def test_get_or_create_metrics(self) -> None:
        """Test getting or creating metrics for an agent."""
        monitor = PerformanceMonitor()
        metrics = monitor.get_or_create_metrics("agent-1")
        assert metrics.agent_id == "agent-1"
        assert "agent-1" in monitor.agents

        # Should return same instance
        metrics2 = monitor.get_or_create_metrics("agent-1")
        assert metrics is metrics2

    @pytest.mark.asyncio
    async def test_record_execution_success(self) -> None:
        """Test recording a successful execution."""
        monitor = PerformanceMonitor()

        async def mock_func(x: int) -> int:
            return x * 2

        result, latency = await monitor.record_execution(
            "agent-1",
            mock_func,
            5,
        )
        assert result == 10
        assert latency > 0
        assert len(monitor.agents["agent-1"].metrics) == 2  # latency + success

    @pytest.mark.asyncio
    async def test_record_execution_failure(self) -> None:
        """Test recording a failed execution."""
        monitor = PerformanceMonitor()

        async def failing_func() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await monitor.record_execution("agent-1", failing_func)

        assert len(monitor.alerts) == 1
        assert "error" in monitor.alerts[0].lower()

    def test_record_accuracy(self) -> None:
        """Test recording accuracy."""
        monitor = PerformanceMonitor()
        monitor.record_accuracy("agent-1", 0.95)
        metrics = monitor.agents["agent-1"]
        assert metrics.get_average(MetricType.ACCURACY) == 0.95

    def test_record_tokens(self) -> None:
        """Test recording token usage."""
        monitor = PerformanceMonitor()
        monitor.record_tokens("agent-1", 150)
        metrics = monitor.agents["agent-1"]
        assert metrics.get_average(MetricType.TOKEN_USAGE) == 150.0

    def test_set_threshold(self) -> None:
        """Test setting alert thresholds."""
        monitor = PerformanceMonitor()
        monitor.set_threshold(MetricType.LATENCY, 1000)
        assert monitor.alert_thresholds[MetricType.LATENCY] == 1000

    def test_latency_alert(self) -> None:
        """Test latency threshold alert."""
        monitor = PerformanceMonitor()
        monitor.set_threshold(MetricType.LATENCY, 100)

        metrics = monitor.get_or_create_metrics("agent-1")
        metrics.record(MetricType.LATENCY, 150)
        monitor._check_thresholds("agent-1", MetricType.LATENCY, 150)

        assert len(monitor.alerts) == 1
        assert "latency" in monitor.alerts[0].lower()

    def test_accuracy_alert(self) -> None:
        """Test accuracy threshold alert."""
        monitor = PerformanceMonitor()
        monitor.set_threshold(MetricType.ACCURACY, 0.9)
        monitor._check_thresholds("agent-1", MetricType.ACCURACY, 0.7)

        assert len(monitor.alerts) == 1
        assert "accuracy" in monitor.alerts[0].lower()

    def test_get_summary(self) -> None:
        """Test getting performance summary."""
        monitor = PerformanceMonitor()
        monitor.record_accuracy("agent-1", 0.95)

        summary = monitor.get_summary("agent-1")
        assert summary is not None
        assert summary.agent_id == "agent-1"

    def test_get_summary_unknown_agent(self) -> None:
        """Test getting summary for unknown agent."""
        monitor = PerformanceMonitor()
        assert monitor.get_summary("unknown") is None

    def test_get_alerts_clear(self) -> None:
        """Test getting and clearing alerts."""
        monitor = PerformanceMonitor()
        monitor.alerts = ["Alert 1", "Alert 2"]

        alerts = monitor.get_alerts(clear=True)
        assert len(alerts) == 2
        assert len(monitor.alerts) == 0


class TestTrajectoryEvaluator:
    """Tests for TrajectoryEvaluator dataclass."""

    def test_evaluator_creation(self) -> None:
        """Test creating a trajectory evaluator."""
        evaluator = TrajectoryEvaluator()
        assert evaluator is not None

    def test_exact_match_success(self) -> None:
        """Test exact match with matching trajectories."""
        evaluator = TrajectoryEvaluator()
        steps = [
            TrajectoryStep(action="parse"),
            TrajectoryStep(action="search"),
            TrajectoryStep(action="format"),
        ]

        result = evaluator.evaluate(steps, steps, TrajectoryMatchType.EXACT)
        assert result.match_score == 1.0
        assert len(result.missing_steps) == 0

    def test_exact_match_failure(self) -> None:
        """Test exact match with different trajectories."""
        evaluator = TrajectoryEvaluator()
        actual = [
            TrajectoryStep(action="parse"),
            TrajectoryStep(action="format"),
        ]
        expected = [
            TrajectoryStep(action="parse"),
            TrajectoryStep(action="search"),
        ]

        match_type = TrajectoryMatchType.EXACT
        result = evaluator.evaluate(actual, expected, match_type)
        assert result.match_score == 0.0

    def test_in_order_match(self) -> None:
        """Test in-order matching."""
        evaluator = TrajectoryEvaluator()
        actual = [
            TrajectoryStep(action="parse"),
            TrajectoryStep(action="validate"),
            TrajectoryStep(action="search"),
            TrajectoryStep(action="format"),
        ]
        expected = [
            TrajectoryStep(action="parse"),
            TrajectoryStep(action="search"),
            TrajectoryStep(action="format"),
        ]

        match_type = TrajectoryMatchType.IN_ORDER
        result = evaluator.evaluate(actual, expected, match_type)
        assert result.match_score == 1.0
        assert "validate" in result.extra_steps

    def test_in_order_partial_match(self) -> None:
        """Test in-order with partial match."""
        evaluator = TrajectoryEvaluator()
        # In-order match: parse matches, format does not come after search
        # so only 1 of 3 expected steps matched in order
        actual = [
            TrajectoryStep(action="parse"),
            TrajectoryStep(action="format"),
        ]
        expected = [
            TrajectoryStep(action="parse"),
            TrajectoryStep(action="search"),
            TrajectoryStep(action="format"),
        ]

        match_type = TrajectoryMatchType.IN_ORDER
        result = evaluator.evaluate(actual, expected, match_type)
        # parse matches, but format comes before search is found
        assert result.match_score == pytest.approx(0.33, rel=0.1)
        assert "search" in result.missing_steps

    def test_any_order_match(self) -> None:
        """Test any-order matching."""
        evaluator = TrajectoryEvaluator()
        actual = [
            TrajectoryStep(action="format"),
            TrajectoryStep(action="parse"),
            TrajectoryStep(action="search"),
        ]
        expected = [
            TrajectoryStep(action="parse"),
            TrajectoryStep(action="search"),
            TrajectoryStep(action="format"),
        ]

        match_type = TrajectoryMatchType.ANY_ORDER
        result = evaluator.evaluate(actual, expected, match_type)
        assert result.match_score == 1.0

    def test_any_order_partial(self) -> None:
        """Test any-order with partial match."""
        evaluator = TrajectoryEvaluator()
        actual = [
            TrajectoryStep(action="parse"),
            TrajectoryStep(action="extra"),
        ]
        expected = [
            TrajectoryStep(action="parse"),
            TrajectoryStep(action="search"),
        ]

        match_type = TrajectoryMatchType.ANY_ORDER
        result = evaluator.evaluate(actual, expected, match_type)
        assert result.match_score == 0.5
        assert "search" in result.missing_steps
        assert "extra" in result.extra_steps

    def test_single_tool_match(self) -> None:
        """Test single tool matching."""
        evaluator = TrajectoryEvaluator()
        actual = [
            TrajectoryStep(action="parse"),
            TrajectoryStep(action="search"),
        ]
        expected = [TrajectoryStep(action="search")]

        match_type = TrajectoryMatchType.SINGLE_TOOL
        result = evaluator.evaluate(actual, expected, match_type)
        assert result.match_score == 1.0

    def test_single_tool_no_match(self) -> None:
        """Test single tool with no match."""
        evaluator = TrajectoryEvaluator()
        actual = [
            TrajectoryStep(action="parse"),
            TrajectoryStep(action="format"),
        ]
        expected = [TrajectoryStep(action="search")]

        match_type = TrajectoryMatchType.SINGLE_TOOL
        result = evaluator.evaluate(actual, expected, match_type)
        assert result.match_score == 0.0

    def test_empty_trajectories(self) -> None:
        """Test with empty expected trajectory."""
        evaluator = TrajectoryEvaluator()
        actual = [TrajectoryStep(action="parse")]
        expected: list[TrajectoryStep] = []

        match_type = TrajectoryMatchType.IN_ORDER
        result = evaluator.evaluate(actual, expected, match_type)
        assert result.match_score == 1.0  # Empty expected = success


class TestLLMJudge:
    """Tests for LLMJudge dataclass."""

    def test_judge_creation(self) -> None:
        """Test creating an LLM judge."""
        judge = LLMJudge()
        assert judge.model_name == "gpt-oss:20b"
        assert "localhost" in judge.base_url

    def test_judge_custom_model(self) -> None:
        """Test judge with custom model."""
        judge = LLMJudge(
            model_name="custom-model", base_url="http://custom:8080"
        )
        assert judge.model_name == "custom-model"

    @pytest.mark.asyncio
    async def test_evaluate(self) -> None:
        """Test evaluating with mocked agent."""
        judge = LLMJudge()

        mock_result = JudgmentResult(
            quality_score=0.9,
            helpfulness_score=0.85,
            accuracy_score=0.95,
            reasoning="Good response",
            improvement_suggestions=[],
        )

        with patch.object(judge, "_get_judge_agent") as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.run.return_value.output = mock_result
            mock_get_agent.return_value = mock_agent

            result = await judge.evaluate(
                "What is Python?",
                "Python is a programming language.",
            )

            assert result.quality_score == 0.9
            assert result.accuracy_score == 0.95

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self) -> None:
        """Test evaluation with context."""
        judge = LLMJudge()

        mock_result = JudgmentResult(
            quality_score=0.8,
            helpfulness_score=0.8,
            accuracy_score=0.9,
            reasoning="Matches context",
            improvement_suggestions=[],
        )

        with patch.object(judge, "_get_judge_agent") as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.run.return_value.output = mock_result
            mock_get_agent.return_value = mock_agent

            result = await judge.evaluate(
                "What is 2+2?",
                "4",
                context="Expected answer is 4",
            )

            assert result.accuracy_score == 0.9


class TestDriftDetector:
    """Tests for DriftDetector dataclass."""

    def test_detector_creation(self) -> None:
        """Test creating a drift detector."""
        detector = DriftDetector()
        assert len(detector.baseline_metrics) == 0
        assert DriftSeverity.LOW in detector.drift_thresholds

    def test_set_baseline(self) -> None:
        """Test setting baseline metrics."""
        detector = DriftDetector()
        metrics = AgentMetrics(agent_id="agent-1")
        metrics.record(MetricType.ACCURACY, 0.95)
        metrics.record(MetricType.LATENCY, 100.0)

        detector.set_baseline("agent-1", metrics)

        assert "agent-1" in detector.baseline_metrics
        assert MetricType.ACCURACY in detector.baseline_metrics["agent-1"]

    def test_detect_drift_no_baseline(self) -> None:
        """Test drift detection without baseline."""
        detector = DriftDetector()
        metrics = AgentMetrics(agent_id="agent-1")

        report = detector.detect_drift("agent-1", metrics)

        assert report.severity == DriftSeverity.NONE
        assert "No baseline" in report.recommendation

    def test_detect_drift_none(self) -> None:
        """Test when no drift is detected."""
        detector = DriftDetector()
        baseline = AgentMetrics(agent_id="agent-1")
        baseline.record(MetricType.ACCURACY, 0.95)

        detector.set_baseline("agent-1", baseline)

        current = AgentMetrics(agent_id="agent-1")
        current.record(MetricType.ACCURACY, 0.94)  # Within threshold

        report = detector.detect_drift("agent-1", current)
        assert report.severity == DriftSeverity.NONE

    def test_detect_drift_low(self) -> None:
        """Test low severity drift detection."""
        detector = DriftDetector()
        baseline = AgentMetrics(agent_id="agent-1")
        baseline.record(MetricType.ACCURACY, 1.0)

        detector.set_baseline("agent-1", baseline)

        current = AgentMetrics(agent_id="agent-1")
        current.record(MetricType.ACCURACY, 0.92)  # 8% drift

        report = detector.detect_drift("agent-1", current)
        assert report.severity == DriftSeverity.LOW

    def test_detect_drift_medium(self) -> None:
        """Test medium severity drift detection."""
        detector = DriftDetector()
        baseline = AgentMetrics(agent_id="agent-1")
        baseline.record(MetricType.ACCURACY, 1.0)

        detector.set_baseline("agent-1", baseline)

        current = AgentMetrics(agent_id="agent-1")
        current.record(MetricType.ACCURACY, 0.80)  # 20% drift

        report = detector.detect_drift("agent-1", current)
        assert report.severity == DriftSeverity.MEDIUM

    def test_detect_drift_high(self) -> None:
        """Test high severity drift detection."""
        detector = DriftDetector()
        baseline = AgentMetrics(agent_id="agent-1")
        baseline.record(MetricType.ACCURACY, 1.0)

        detector.set_baseline("agent-1", baseline)

        current = AgentMetrics(agent_id="agent-1")
        current.record(MetricType.ACCURACY, 0.70)  # 30% drift

        report = detector.detect_drift("agent-1", current)
        assert report.severity == DriftSeverity.HIGH

    def test_detect_drift_critical(self) -> None:
        """Test critical severity drift detection."""
        detector = DriftDetector()
        baseline = AgentMetrics(agent_id="agent-1")
        baseline.record(MetricType.ACCURACY, 1.0)

        detector.set_baseline("agent-1", baseline)

        current = AgentMetrics(agent_id="agent-1")
        current.record(MetricType.ACCURACY, 0.50)  # 50% drift

        report = detector.detect_drift("agent-1", current)
        assert report.severity == DriftSeverity.CRITICAL

    def test_detect_drift_zero_baseline(self) -> None:
        """Test drift detection with zero baseline."""
        detector = DriftDetector()
        baseline = AgentMetrics(agent_id="agent-1")
        baseline.record(MetricType.TOKEN_USAGE, 0.0)

        detector.set_baseline("agent-1", baseline)

        current = AgentMetrics(agent_id="agent-1")
        current.record(MetricType.TOKEN_USAGE, 100.0)

        # Should handle zero baseline gracefully
        report = detector.detect_drift("agent-1", current)
        assert report is not None


class TestABTestRunner:
    """Tests for ABTestRunner dataclass."""

    def test_runner_creation(self) -> None:
        """Test creating an A/B test runner."""
        runner = ABTestRunner()
        assert len(runner.results_a) == 0
        assert len(runner.results_b) == 0

    def test_runner_custom_names(self) -> None:
        """Test runner with custom variant names."""
        runner = ABTestRunner(
            variant_a_name="Control",
            variant_b_name="Treatment",
        )
        assert runner.variant_a_name == "Control"
        assert runner.variant_b_name == "Treatment"

    def test_record_result_a(self) -> None:
        """Test recording result for variant A."""
        runner = ABTestRunner()
        runner.record_result("a", 0.9)
        assert len(runner.results_a) == 1
        assert runner.results_a[0] == 0.9

    def test_record_result_b(self) -> None:
        """Test recording result for variant B."""
        runner = ABTestRunner()
        runner.record_result("b", 0.85)
        assert len(runner.results_b) == 1

    def test_record_result_by_name(self) -> None:
        """Test recording result by variant name."""
        runner = ABTestRunner(variant_a_name="Control")
        runner.record_result("Control", 0.9)
        assert len(runner.results_a) == 1

    def test_get_results_insufficient_data(self) -> None:
        """Test getting results with insufficient data."""
        runner = ABTestRunner()
        result = runner.get_results()
        assert result.winner is None
        assert result.sample_size == 0
        assert "Insufficient" in result.details

    def test_get_results_clear_winner(self) -> None:
        """Test getting results with clear winner."""
        runner = ABTestRunner()
        for _ in range(20):
            runner.record_result("a", 0.8)
            runner.record_result("b", 0.95)

        result = runner.get_results()
        assert result.variant_b_score > result.variant_a_score
        assert result.sample_size == 40

    def test_get_results_no_winner(self) -> None:
        """Test results with no clear winner."""
        runner = ABTestRunner()
        for _ in range(5):
            runner.record_result("a", 0.85)
            runner.record_result("b", 0.86)

        result = runner.get_results()
        # Low sample size and small difference = low confidence
        assert result.confidence < 0.9

    def test_reset(self) -> None:
        """Test resetting test data."""
        runner = ABTestRunner()
        runner.record_result("a", 0.9)
        runner.record_result("b", 0.8)
        runner.reset()
        assert len(runner.results_a) == 0
        assert len(runner.results_b) == 0


class TestEvaluateResponseAccuracy:
    """Tests for evaluate_response_accuracy function."""

    def test_exact_match(self) -> None:
        """Test exact match evaluation."""
        result = evaluate_response_accuracy("Hello", "Hello")
        assert result.score == 1.0
        assert result.passed is True

    def test_case_insensitive_match(self) -> None:
        """Test case-insensitive matching."""
        result = evaluate_response_accuracy("HELLO", "hello")
        assert result.score == 1.0
        assert result.passed is True

    def test_case_sensitive_mismatch(self) -> None:
        """Test case-sensitive matching fails on case diff."""
        result = evaluate_response_accuracy(
            "HELLO",
            "hello",
            case_sensitive=True,
        )
        assert result.score == 0.0
        assert result.passed is False

    def test_whitespace_trimmed(self) -> None:
        """Test whitespace is trimmed."""
        result = evaluate_response_accuracy("  Hello  ", "Hello")
        assert result.score == 1.0

    def test_mismatch(self) -> None:
        """Test mismatched responses."""
        result = evaluate_response_accuracy("Hello", "Goodbye")
        assert result.score == 0.0
        assert result.passed is False
        assert "differs" in result.details


class TestCalculatePrecisionRecall:
    """Tests for calculate_precision_recall function."""

    def test_perfect_match(self) -> None:
        """Test perfect precision and recall."""
        precision, recall, f1 = calculate_precision_recall(
            ["a", "b", "c"],
            ["a", "b", "c"],
        )
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_partial_match(self) -> None:
        """Test partial match."""
        precision, recall, f1 = calculate_precision_recall(
            ["a", "b", "extra"],
            ["a", "b", "c"],
        )
        # 2 correct out of 3 predicted
        assert precision == pytest.approx(0.67, rel=0.1)
        # 2 correct out of 3 expected
        assert recall == pytest.approx(0.67, rel=0.1)

    def test_no_match(self) -> None:
        """Test no matching items."""
        precision, recall, f1 = calculate_precision_recall(
            ["x", "y"],
            ["a", "b"],
        )
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_empty_predicted(self) -> None:
        """Test with empty predictions."""
        precision, recall, f1 = calculate_precision_recall([], ["a", "b"])
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_empty_ground_truth(self) -> None:
        """Test with empty ground truth."""
        precision, recall, f1 = calculate_precision_recall(["a", "b"], [])
        assert precision == 0.0
        assert recall == 0.0


class TestGenerateEvaluationReport:
    """Tests for generate_evaluation_report function."""

    def test_minimal_report(self) -> None:
        """Test generating minimal report."""
        report = generate_evaluation_report("agent-1")
        assert report.agent_id == "agent-1"
        assert report.performance_summary is None
        assert len(report.recommendations) == 0

    def test_report_with_metrics(self) -> None:
        """Test report with performance metrics."""
        metrics = AgentMetrics(agent_id="agent-1")
        metrics.record(MetricType.ACCURACY, 0.95)

        report = generate_evaluation_report("agent-1", metrics=metrics)
        assert report.performance_summary is not None

    def test_report_with_drift(self) -> None:
        """Test report with drift detection."""
        drift = DriftReport(
            severity=DriftSeverity.MEDIUM,
            affected_metrics=[MetricType.ACCURACY],
            baseline_values={"accuracy": 0.95},
            current_values={"accuracy": 0.80},
            drift_percentage=15.0,
            recommendation="Investigate accuracy drop",
        )

        report = generate_evaluation_report("agent-1", drift_report=drift)
        assert drift.recommendation in report.recommendations

    def test_report_with_low_trajectories(self) -> None:
        """Test report flags low-scoring trajectories."""
        trajectories = [
            TrajectoryResult(
                match_type=TrajectoryMatchType.IN_ORDER,
                match_score=0.5,
                matched_steps=[],
                missing_steps=["step1"],
                extra_steps=[],
                details="Low score",
            ),
        ]

        report = generate_evaluation_report(
            "agent-1",
            trajectory_results=trajectories,
        )
        assert any("trajectories" in r.lower() for r in report.recommendations)

    def test_report_with_ab_winner(self) -> None:
        """Test report with A/B test winner."""
        ab_results = [
            ABTestResult(
                variant_a_name="A",
                variant_b_name="B",
                winner="B",
                variant_a_score=0.8,
                variant_b_score=0.9,
                confidence=0.95,
                sample_size=100,
                details="B wins",
            ),
        ]

        report = generate_evaluation_report("agent-1", ab_results=ab_results)
        assert any("B" in r for r in report.recommendations)


class TestIntegrationScenarios:
    """Integration tests for evaluation scenarios."""

    @pytest.mark.asyncio
    async def test_complete_evaluation_workflow(self) -> None:
        """Test complete evaluation workflow."""
        # Setup monitor
        monitor = PerformanceMonitor()
        monitor.set_threshold(MetricType.LATENCY, 5000)

        # Record some executions
        async def mock_agent(x: str) -> str:
            return f"Response: {x}"

        for query in ["query1", "query2", "query3"]:
            await monitor.record_execution("agent-1", mock_agent, query)
            monitor.record_accuracy("agent-1", 0.9)
            monitor.record_tokens("agent-1", 100)

        # Setup drift detection
        detector = DriftDetector()
        baseline = monitor.agents["agent-1"]
        detector.set_baseline("agent-1", baseline)

        # Simulate degraded performance
        degraded = AgentMetrics(agent_id="agent-1")
        degraded.record(MetricType.ACCURACY, 0.70)
        degraded.record(MetricType.LATENCY, 500.0)

        drift = detector.detect_drift("agent-1", degraded)

        # Generate report
        report = generate_evaluation_report(
            "agent-1",
            metrics=degraded,
            drift_report=drift,
        )

        assert report.agent_id == "agent-1"
        assert report.drift_report is not None

    def test_trajectory_evaluation_pipeline(self) -> None:
        """Test trajectory evaluation pipeline."""
        evaluator = TrajectoryEvaluator()

        expected = [
            TrajectoryStep(action="parse", tool_name="parser"),
            TrajectoryStep(action="search", tool_name="db"),
            TrajectoryStep(action="format", tool_name="formatter"),
        ]

        # Test different actual trajectories
        test_cases = [
            (
                [
                    TrajectoryStep(action="parse"),
                    TrajectoryStep(action="search"),
                    TrajectoryStep(action="format"),
                ],
                TrajectoryMatchType.EXACT,
                1.0,
            ),
            (
                [
                    TrajectoryStep(action="parse"),
                    TrajectoryStep(action="validate"),
                    TrajectoryStep(action="search"),
                    TrajectoryStep(action="format"),
                ],
                TrajectoryMatchType.IN_ORDER,
                1.0,
            ),
            (
                [
                    TrajectoryStep(action="format"),
                    TrajectoryStep(action="search"),
                    TrajectoryStep(action="parse"),
                ],
                TrajectoryMatchType.ANY_ORDER,
                1.0,
            ),
        ]

        for actual, match_type, expected_score in test_cases:
            result = evaluator.evaluate(actual, expected, match_type)
            assert result.match_score == expected_score

    def test_ab_test_workflow(self) -> None:
        """Test complete A/B test workflow."""
        runner = ABTestRunner(
            variant_a_name="Prompt v1",
            variant_b_name="Prompt v2",
        )

        # Simulate running test
        import random

        random.seed(42)

        for _ in range(30):
            runner.record_result("a", 0.80 + random.random() * 0.1)
            runner.record_result("b", 0.85 + random.random() * 0.1)

        result = runner.get_results()

        assert result.sample_size == 60
        assert result.variant_b_score > result.variant_a_score
