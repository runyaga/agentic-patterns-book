"""
Evaluation and Monitoring pattern implementation.

This module provides tools for systematically assessing agent performance,
monitoring progress toward goals, and detecting operational anomalies.

Key components:
- AgentMetrics: Track accuracy, latency, and resource consumption
- PerformanceMonitor: Real-time performance monitoring
- TrajectoryEvaluator: Compare agent actions to expected paths
- LLMJudge: Qualitative assessment using an LLM evaluator
- DriftDetector: Detect performance degradation over time
- ABTestRunner: Compare agent versions systematically

Example usage:
    monitor = PerformanceMonitor()
    result = await monitor.record_execution(
        "agent-1",
        async_func,
        "user query"
    )
    print(monitor.get_summary("agent-1"))
"""

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent

from agentic_patterns import get_model


class MetricType(str, Enum):
    """Types of metrics that can be tracked."""

    ACCURACY = "accuracy"
    LATENCY = "latency"
    TOKEN_USAGE = "token_usage"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


class TrajectoryMatchType(str, Enum):
    """Methods for comparing agent trajectories."""

    EXACT = "exact"  # Perfect match required
    IN_ORDER = "in_order"  # Correct actions in order, extra allowed
    ANY_ORDER = "any_order"  # Correct actions any order, extra allowed
    SINGLE_TOOL = "single_tool"  # Check for specific action presence


class DriftSeverity(str, Enum):
    """Severity levels for drift detection."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------


class MetricValue(BaseModel):
    """A single metric measurement."""

    metric_type: MetricType
    value: float
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Result of evaluating an agent response."""

    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    details: str
    metric_type: MetricType = MetricType.ACCURACY
    timestamp: datetime = Field(default_factory=datetime.now)


class TrajectoryStep(BaseModel):
    """A single step in an agent's trajectory."""

    action: str
    tool_name: str | None = None
    input_data: str | None = None
    output_data: str | None = None
    duration_ms: float = 0.0


class TrajectoryResult(BaseModel):
    """Result of trajectory comparison."""

    match_type: TrajectoryMatchType
    match_score: float = Field(ge=0.0, le=1.0)
    matched_steps: list[str]
    missing_steps: list[str]
    extra_steps: list[str]
    details: str


class JudgmentResult(BaseModel):
    """Result from LLM-as-Judge evaluation."""

    quality_score: float = Field(ge=0.0, le=1.0)
    helpfulness_score: float = Field(ge=0.0, le=1.0)
    accuracy_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    improvement_suggestions: list[str] = Field(default_factory=list)


class DriftReport(BaseModel):
    """Report on detected performance drift."""

    severity: DriftSeverity
    affected_metrics: list[MetricType]
    baseline_values: dict[str, float]
    current_values: dict[str, float]
    drift_percentage: float
    recommendation: str


class ABTestResult(BaseModel):
    """Result of A/B test comparison."""

    variant_a_name: str
    variant_b_name: str
    winner: str | None
    variant_a_score: float
    variant_b_score: float
    confidence: float
    sample_size: int
    details: str


class PerformanceSummary(BaseModel):
    """Summary of agent performance metrics."""

    agent_id: str
    total_executions: int
    avg_latency_ms: float
    avg_accuracy: float
    success_rate: float
    total_tokens: int
    period_start: datetime
    period_end: datetime


class EvaluationReport(BaseModel):
    """Comprehensive evaluation report."""

    agent_id: str
    report_timestamp: datetime = Field(default_factory=datetime.now)
    performance_summary: PerformanceSummary | None = None
    trajectory_results: list[TrajectoryResult] = Field(default_factory=list)
    drift_report: DriftReport | None = None
    ab_test_results: list[ABTestResult] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Dataclasses for runtime state
# ---------------------------------------------------------------------


@dataclass
class AgentMetrics:
    """
    Track performance metrics for an agent.

    Collects and aggregates metrics like accuracy, latency, and token usage.
    """

    agent_id: str
    metrics: list[MetricValue] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    def record(
        self,
        metric_type: MetricType,
        value: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a metric value."""
        self.metrics.append(
            MetricValue(
                metric_type=metric_type,
                value=value,
                metadata=metadata or {},
            )
        )

    def get_average(self, metric_type: MetricType) -> float | None:
        """Get average value for a metric type."""
        values = [
            m.value for m in self.metrics if m.metric_type == metric_type
        ]
        if not values:
            return None
        return sum(values) / len(values)

    def get_recent(
        self,
        metric_type: MetricType,
        count: int = 10,
    ) -> list[float]:
        """Get most recent values for a metric type."""
        values = [
            m.value for m in self.metrics if m.metric_type == metric_type
        ]
        return values[-count:]

    def get_summary(self) -> PerformanceSummary:
        """Generate performance summary."""
        lat_type = MetricType.LATENCY
        acc_type = MetricType.ACCURACY
        tok_type = MetricType.TOKEN_USAGE
        suc_type = MetricType.SUCCESS_RATE

        latencies = [
            m.value for m in self.metrics if m.metric_type == lat_type
        ]
        accuracies = [
            m.value for m in self.metrics if m.metric_type == acc_type
        ]
        tokens = [m.value for m in self.metrics if m.metric_type == tok_type]
        success_count = len(
            [m for m in self.metrics if m.metric_type == suc_type]
        )

        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0

        return PerformanceSummary(
            agent_id=self.agent_id,
            total_executions=len(self.metrics),
            avg_latency_ms=avg_lat,
            avg_accuracy=avg_acc,
            success_rate=(
                success_count / len(self.metrics) if self.metrics else 0
            ),
            total_tokens=int(sum(tokens)),
            period_start=self.start_time,
            period_end=datetime.now(),
        )


@dataclass
class PerformanceMonitor:
    """
    Real-time performance monitoring for agents.

    Tracks execution metrics and provides live performance dashboards.
    """

    agents: dict[str, AgentMetrics] = field(default_factory=dict)
    alert_thresholds: dict[MetricType, float] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)

    def get_or_create_metrics(self, agent_id: str) -> AgentMetrics:
        """Get or create metrics tracker for an agent."""
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentMetrics(agent_id=agent_id)
        return self.agents[agent_id]

    async def record_execution(
        self,
        agent_id: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, float]:
        """
        Record execution metrics for a function call.

        Returns:
            Tuple of (result, latency_ms)
        """
        metrics = self.get_or_create_metrics(agent_id)
        start = datetime.now()

        try:
            result = await func(*args, **kwargs)
            latency_ms = (datetime.now() - start).total_seconds() * 1000
            metrics.record(MetricType.LATENCY, latency_ms)
            metrics.record(MetricType.SUCCESS_RATE, 1.0)
            self._check_thresholds(agent_id, MetricType.LATENCY, latency_ms)
            return result, latency_ms
        except Exception as e:
            latency_ms = (datetime.now() - start).total_seconds() * 1000
            metrics.record(MetricType.LATENCY, latency_ms)
            metrics.record(MetricType.ERROR_RATE, 1.0)
            self.alerts.append(f"Agent {agent_id} error: {e}")
            raise

    def record_accuracy(
        self,
        agent_id: str,
        score: float,
    ) -> None:
        """Record an accuracy measurement."""
        metrics = self.get_or_create_metrics(agent_id)
        metrics.record(MetricType.ACCURACY, score)
        self._check_thresholds(agent_id, MetricType.ACCURACY, score)

    def record_tokens(
        self,
        agent_id: str,
        token_count: int,
    ) -> None:
        """Record token usage."""
        metrics = self.get_or_create_metrics(agent_id)
        metrics.record(MetricType.TOKEN_USAGE, float(token_count))

    def set_threshold(
        self,
        metric_type: MetricType,
        threshold: float,
    ) -> None:
        """Set alert threshold for a metric type."""
        self.alert_thresholds[metric_type] = threshold

    def _check_thresholds(
        self,
        agent_id: str,
        metric_type: MetricType,
        value: float,
    ) -> None:
        """Check if value exceeds threshold and create alert."""
        if metric_type not in self.alert_thresholds:
            return

        threshold = self.alert_thresholds[metric_type]
        if metric_type == MetricType.LATENCY and value > threshold:
            self.alerts.append(
                f"Agent {agent_id}: latency {value:.1f}ms > {threshold}ms"
            )
        elif metric_type == MetricType.ACCURACY and value < threshold:
            self.alerts.append(
                f"Agent {agent_id}: accuracy {value:.2f} < {threshold}"
            )

    def get_summary(self, agent_id: str) -> PerformanceSummary | None:
        """Get performance summary for an agent."""
        if agent_id not in self.agents:
            return None
        return self.agents[agent_id].get_summary()

    def get_alerts(self, clear: bool = False) -> list[str]:
        """Get all alerts, optionally clearing them."""
        alerts = list(self.alerts)
        if clear:
            self.alerts.clear()
        return alerts


@dataclass
class TrajectoryEvaluator:
    """
    Evaluate agent trajectories against expected paths.

    Compares the sequence of actions an agent takes to an ideal trajectory.
    """

    def evaluate(
        self,
        actual: list[TrajectoryStep],
        expected: list[TrajectoryStep],
        match_type: TrajectoryMatchType = TrajectoryMatchType.IN_ORDER,
    ) -> TrajectoryResult:
        """
        Evaluate actual trajectory against expected.

        Args:
            actual: The actual steps the agent took.
            expected: The expected/ideal steps.
            match_type: How strictly to compare.

        Returns:
            TrajectoryResult with match analysis.
        """
        actual_actions = [s.action for s in actual]
        expected_actions = [s.action for s in expected]

        if match_type == TrajectoryMatchType.EXACT:
            return self._exact_match(actual_actions, expected_actions)
        elif match_type == TrajectoryMatchType.IN_ORDER:
            return self._in_order_match(actual_actions, expected_actions)
        elif match_type == TrajectoryMatchType.ANY_ORDER:
            return self._any_order_match(actual_actions, expected_actions)
        else:  # SINGLE_TOOL
            return self._single_tool_match(actual_actions, expected_actions)

    def _exact_match(
        self,
        actual: list[str],
        expected: list[str],
    ) -> TrajectoryResult:
        """Check for exact sequence match."""
        is_match = actual == expected
        matched = actual if is_match else []
        missing = expected if not is_match else []
        extra = [] if is_match else [a for a in actual if a not in expected]

        return TrajectoryResult(
            match_type=TrajectoryMatchType.EXACT,
            match_score=1.0 if is_match else 0.0,
            matched_steps=matched,
            missing_steps=missing,
            extra_steps=extra,
            details="Exact match" if is_match else "Trajectories differ",
        )

    def _in_order_match(
        self,
        actual: list[str],
        expected: list[str],
    ) -> TrajectoryResult:
        """Check if expected actions appear in order (extras allowed)."""
        matched = []
        exp_idx = 0

        for action in actual:
            if exp_idx < len(expected) and action == expected[exp_idx]:
                matched.append(action)
                exp_idx += 1

        missing = expected[exp_idx:]
        extra = [a for a in actual if a not in expected]
        score = len(matched) / len(expected) if expected else 1.0

        return TrajectoryResult(
            match_type=TrajectoryMatchType.IN_ORDER,
            match_score=score,
            matched_steps=matched,
            missing_steps=missing,
            extra_steps=extra,
            details=f"Matched {len(matched)}/{len(expected)} in order",
        )

    def _any_order_match(
        self,
        actual: list[str],
        expected: list[str],
    ) -> TrajectoryResult:
        """Check if expected actions appear in any order."""
        expected_set = set(expected)
        actual_set = set(actual)

        matched = list(expected_set & actual_set)
        missing = list(expected_set - actual_set)
        extra = list(actual_set - expected_set)
        score = len(matched) / len(expected) if expected else 1.0

        return TrajectoryResult(
            match_type=TrajectoryMatchType.ANY_ORDER,
            match_score=score,
            matched_steps=matched,
            missing_steps=missing,
            extra_steps=extra,
            details=f"Matched {len(matched)}/{len(expected)} actions",
        )

    def _single_tool_match(
        self,
        actual: list[str],
        expected: list[str],
    ) -> TrajectoryResult:
        """Check if at least one expected action is present."""
        expected_set = set(expected)
        actual_set = set(actual)

        matched = list(expected_set & actual_set)
        is_match = len(matched) > 0

        return TrajectoryResult(
            match_type=TrajectoryMatchType.SINGLE_TOOL,
            match_score=1.0 if is_match else 0.0,
            matched_steps=matched,
            missing_steps=list(expected_set - actual_set),
            extra_steps=[],
            details="Required tool found" if is_match else "No required tool",
        )


@dataclass
class LLMJudge:
    """
    Use an LLM to evaluate response quality.

    Provides nuanced, qualitative assessment of agent outputs.
    """

    model_name: str = "gpt-oss:20b"
    base_url: str = "http://localhost:11434/v1"

    def _get_judge_agent(self) -> Agent[None, JudgmentResult]:
        """Create the judge agent."""
        model = get_model(
            model_name=self.model_name,
            base_url=self.base_url,
        )

        return Agent(
            model,
            output_type=JudgmentResult,
            system_prompt="""You are an expert evaluator of AI agent responses.
Evaluate the given response for:
1. Quality: Overall response quality (0-1)
2. Helpfulness: How helpful is the response (0-1)
3. Accuracy: Factual correctness if verifiable (0-1)

Provide reasoning and improvement suggestions.""",
        )

    async def evaluate(
        self,
        query: str,
        response: str,
        context: str | None = None,
    ) -> JudgmentResult:
        """
        Evaluate a response using LLM-as-Judge.

        Args:
            query: The original query.
            response: The agent's response.
            context: Optional context about expected answer.

        Returns:
            JudgmentResult with scores and reasoning.
        """
        agent = self._get_judge_agent()
        prompt = f"Query: {query}\n\nResponse: {response}"
        if context:
            prompt += f"\n\nContext/Expected: {context}"

        result = await agent.run(prompt)
        return result.output


@dataclass
class DriftDetector:
    """
    Detect performance drift over time.

    Monitors for degradation in agent performance metrics.
    """

    baseline_metrics: dict[str, dict[MetricType, float]] = field(
        default_factory=dict
    )
    drift_thresholds: dict[DriftSeverity, float] = field(
        default_factory=lambda: {
            DriftSeverity.LOW: 0.05,  # 5% deviation
            DriftSeverity.MEDIUM: 0.15,  # 15% deviation
            DriftSeverity.HIGH: 0.25,  # 25% deviation
            DriftSeverity.CRITICAL: 0.40,  # 40% deviation
        }
    )

    def set_baseline(
        self,
        agent_id: str,
        metrics: AgentMetrics,
    ) -> None:
        """Set baseline metrics for drift comparison."""
        baselines: dict[MetricType, float] = {}

        for metric_type in MetricType:
            avg = metrics.get_average(metric_type)
            if avg is not None:
                baselines[metric_type] = avg

        self.baseline_metrics[agent_id] = baselines

    def detect_drift(
        self,
        agent_id: str,
        current_metrics: AgentMetrics,
    ) -> DriftReport:
        """
        Detect drift between baseline and current metrics.

        Args:
            agent_id: The agent to check.
            current_metrics: Current metric values.

        Returns:
            DriftReport with severity and details.
        """
        if agent_id not in self.baseline_metrics:
            return DriftReport(
                severity=DriftSeverity.NONE,
                affected_metrics=[],
                baseline_values={},
                current_values={},
                drift_percentage=0.0,
                recommendation="No baseline set for comparison",
            )

        baselines = self.baseline_metrics[agent_id]
        affected: list[MetricType] = []
        max_drift = 0.0
        baseline_vals: dict[str, float] = {}
        current_vals: dict[str, float] = {}

        for metric_type, baseline_val in baselines.items():
            current_avg = current_metrics.get_average(metric_type)
            if current_avg is None:
                continue

            baseline_vals[metric_type.value] = baseline_val
            current_vals[metric_type.value] = current_avg

            if baseline_val != 0:
                drift = abs(current_avg - baseline_val) / baseline_val
            else:
                drift = abs(current_avg) if current_avg != 0 else 0

            if drift > max_drift:
                max_drift = drift

            if drift > self.drift_thresholds[DriftSeverity.LOW]:
                affected.append(metric_type)

        severity = self._get_severity(max_drift)
        recommendation = self._get_recommendation(severity, affected)

        return DriftReport(
            severity=severity,
            affected_metrics=affected,
            baseline_values=baseline_vals,
            current_values=current_vals,
            drift_percentage=max_drift * 100,
            recommendation=recommendation,
        )

    def _get_severity(self, drift: float) -> DriftSeverity:
        """Determine severity based on drift amount."""
        if drift >= self.drift_thresholds[DriftSeverity.CRITICAL]:
            return DriftSeverity.CRITICAL
        elif drift >= self.drift_thresholds[DriftSeverity.HIGH]:
            return DriftSeverity.HIGH
        elif drift >= self.drift_thresholds[DriftSeverity.MEDIUM]:
            return DriftSeverity.MEDIUM
        elif drift >= self.drift_thresholds[DriftSeverity.LOW]:
            return DriftSeverity.LOW
        return DriftSeverity.NONE

    def _get_recommendation(
        self,
        severity: DriftSeverity,
        affected: list[MetricType],
    ) -> str:
        """Generate recommendation based on drift severity."""
        if severity == DriftSeverity.NONE:
            return "No significant drift detected"
        elif severity == DriftSeverity.LOW:
            return "Minor drift - continue monitoring"
        elif severity == DriftSeverity.MEDIUM:
            affected_str = ", ".join(m.value for m in affected)
            return f"Notable drift in {affected_str} - investigate causes"
        elif severity == DriftSeverity.HIGH:
            return "Significant drift - review model/prompts, retrain"
        else:  # CRITICAL
            return "Critical drift - immediate intervention required"


@dataclass
class ABTestRunner:
    """
    Run A/B tests comparing agent variants.

    Systematically compares different agent versions or strategies.
    """

    results_a: list[float] = field(default_factory=list)
    results_b: list[float] = field(default_factory=list)
    variant_a_name: str = "Variant A"
    variant_b_name: str = "Variant B"

    def record_result(self, variant: str, score: float) -> None:
        """Record a result for a variant."""
        if variant == "a" or variant == self.variant_a_name:
            self.results_a.append(score)
        else:
            self.results_b.append(score)

    def get_results(self) -> ABTestResult:
        """
        Calculate A/B test results.

        Returns:
            ABTestResult with winner and statistics.
        """
        if not self.results_a or not self.results_b:
            return ABTestResult(
                variant_a_name=self.variant_a_name,
                variant_b_name=self.variant_b_name,
                winner=None,
                variant_a_score=0.0,
                variant_b_score=0.0,
                confidence=0.0,
                sample_size=0,
                details="Insufficient data",
            )

        avg_a = sum(self.results_a) / len(self.results_a)
        avg_b = sum(self.results_b) / len(self.results_b)
        total_samples = len(self.results_a) + len(self.results_b)

        # Simple confidence based on sample size and difference
        diff = abs(avg_a - avg_b)
        min_samples = min(len(self.results_a), len(self.results_b))
        confidence = min(0.95, diff * min_samples / 10) if diff > 0 else 0.5

        winner = None
        if confidence >= 0.8:
            if avg_a > avg_b:
                winner = self.variant_a_name
            else:
                winner = self.variant_b_name

        return ABTestResult(
            variant_a_name=self.variant_a_name,
            variant_b_name=self.variant_b_name,
            winner=winner,
            variant_a_score=avg_a,
            variant_b_score=avg_b,
            confidence=confidence,
            sample_size=total_samples,
            details=f"A: {avg_a:.3f} (n={len(self.results_a)}), "
            f"B: {avg_b:.3f} (n={len(self.results_b)})",
        )

    def reset(self) -> None:
        """Reset test data."""
        self.results_a.clear()
        self.results_b.clear()


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def evaluate_response_accuracy(
    actual: str,
    expected: str,
    case_sensitive: bool = False,
) -> EvaluationResult:
    """
    Evaluate response accuracy with exact match.

    Args:
        actual: The actual response.
        expected: The expected response.
        case_sensitive: Whether comparison is case-sensitive.

    Returns:
        EvaluationResult with score and details.
    """
    if case_sensitive:
        is_match = actual.strip() == expected.strip()
    else:
        is_match = actual.strip().lower() == expected.strip().lower()

    details = "Exact match" if is_match else "Response differs from expected"
    return EvaluationResult(
        score=1.0 if is_match else 0.0,
        passed=is_match,
        details=details,
        metric_type=MetricType.ACCURACY,
    )


def calculate_precision_recall(
    predicted: list[str],
    ground_truth: list[str],
) -> tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.

    Args:
        predicted: Predicted actions/outputs.
        ground_truth: Expected actions/outputs.

    Returns:
        Tuple of (precision, recall, f1_score).
    """
    predicted_set = set(predicted)
    truth_set = set(ground_truth)

    true_positives = len(predicted_set & truth_set)

    precision = true_positives / len(predicted_set) if predicted_set else 0.0
    recall = true_positives / len(truth_set) if truth_set else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def generate_evaluation_report(
    agent_id: str,
    metrics: AgentMetrics | None = None,
    trajectory_results: list[TrajectoryResult] | None = None,
    drift_report: DriftReport | None = None,
    ab_results: list[ABTestResult] | None = None,
) -> EvaluationReport:
    """
    Generate a comprehensive evaluation report.

    Args:
        agent_id: The agent being evaluated.
        metrics: Optional performance metrics.
        trajectory_results: Optional trajectory evaluations.
        drift_report: Optional drift detection results.
        ab_results: Optional A/B test results.

    Returns:
        EvaluationReport with all findings.
    """
    recommendations: list[str] = []

    # Add recommendations based on findings
    if drift_report and drift_report.severity != DriftSeverity.NONE:
        recommendations.append(drift_report.recommendation)

    if trajectory_results:
        low_scores = [t for t in trajectory_results if t.match_score < 0.8]
        if low_scores:
            recommendations.append(
                f"Review {len(low_scores)} low-scoring trajectories"
            )

    if ab_results:
        for ab in ab_results:
            if ab.winner:
                recommendations.append(
                    f"Consider adopting {ab.winner} based on A/B test"
                )

    return EvaluationReport(
        agent_id=agent_id,
        performance_summary=metrics.get_summary() if metrics else None,
        trajectory_results=trajectory_results or [],
        drift_report=drift_report,
        ab_test_results=ab_results or [],
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------


if __name__ == "__main__":
    import asyncio

    async def demo() -> None:
        """Demonstrate evaluation and monitoring capabilities."""
        print("=" * 60)
        print("Evaluation and Monitoring Pattern Demo")
        print("=" * 60)

        # 1. Performance monitoring
        print("\n--- Performance Monitoring ---")
        monitor = PerformanceMonitor()
        monitor.set_threshold(MetricType.LATENCY, 1000)
        monitor.set_threshold(MetricType.ACCURACY, 0.8)

        async def mock_agent_call(query: str) -> str:
            await asyncio.sleep(0.1)  # Simulate work
            return f"Response to: {query}"

        _, latency = await monitor.record_execution(
            "agent-1",
            mock_agent_call,
            "What is the weather?",
        )
        print(f"Recorded execution with latency: {latency:.1f}ms")

        monitor.record_accuracy("agent-1", 0.95)
        monitor.record_tokens("agent-1", 150)

        summary = monitor.get_summary("agent-1")
        if summary:
            print(f"Agent: {summary.agent_id}")
            print(f"  Executions: {summary.total_executions}")
            print(f"  Avg Latency: {summary.avg_latency_ms:.1f}ms")
            print(f"  Avg Accuracy: {summary.avg_accuracy:.2f}")

        # 2. Trajectory evaluation
        print("\n--- Trajectory Evaluation ---")
        evaluator = TrajectoryEvaluator()

        expected = [
            TrajectoryStep(action="parse_query", tool_name="parser"),
            TrajectoryStep(action="search_db", tool_name="database"),
            TrajectoryStep(action="format_response", tool_name="formatter"),
        ]
        actual = [
            TrajectoryStep(action="parse_query", tool_name="parser"),
            TrajectoryStep(action="validate_input", tool_name="validator"),
            TrajectoryStep(action="search_db", tool_name="database"),
            TrajectoryStep(action="format_response", tool_name="formatter"),
        ]

        match_type = TrajectoryMatchType.IN_ORDER
        result = evaluator.evaluate(actual, expected, match_type)
        print(f"Trajectory Match: {result.match_type.value}")
        print(f"  Score: {result.match_score:.2f}")
        print(f"  Matched: {result.matched_steps}")
        print(f"  Extra: {result.extra_steps}")

        # 3. Drift detection
        print("\n--- Drift Detection ---")
        detector = DriftDetector()

        baseline_metrics = AgentMetrics(agent_id="agent-1")
        baseline_metrics.record(MetricType.ACCURACY, 0.95)
        baseline_metrics.record(MetricType.ACCURACY, 0.93)
        baseline_metrics.record(MetricType.LATENCY, 100)
        baseline_metrics.record(MetricType.LATENCY, 110)
        detector.set_baseline("agent-1", baseline_metrics)

        current_metrics = AgentMetrics(agent_id="agent-1")
        current_metrics.record(MetricType.ACCURACY, 0.80)
        current_metrics.record(MetricType.ACCURACY, 0.78)
        current_metrics.record(MetricType.LATENCY, 150)
        current_metrics.record(MetricType.LATENCY, 160)

        drift = detector.detect_drift("agent-1", current_metrics)
        print(f"Drift Severity: {drift.severity.value}")
        print(f"  Affected: {[m.value for m in drift.affected_metrics]}")
        print(f"  Drift %: {drift.drift_percentage:.1f}%")
        print(f"  Recommendation: {drift.recommendation}")

        # 4. A/B testing
        print("\n--- A/B Testing ---")
        ab_test = ABTestRunner(
            variant_a_name="GPT-4 Prompt",
            variant_b_name="Claude Prompt",
        )

        import random as rand_mod

        for _ in range(10):
            noise_a = 0.1 * (0.5 - rand_mod.random())
            noise_b = 0.1 * (0.5 - rand_mod.random())
            ab_test.record_result("a", 0.85 + noise_a)
            ab_test.record_result("b", 0.90 + noise_b)

        ab_result = ab_test.get_results()
        a_name = ab_result.variant_a_name
        b_name = ab_result.variant_b_name
        print(f"A/B Test: {a_name} vs {b_name}")
        print(f"  Winner: {ab_result.winner or 'No clear winner'}")
        print(f"  Confidence: {ab_result.confidence:.2f}")
        print(f"  Details: {ab_result.details}")

        # 5. Generate report
        print("\n--- Evaluation Report ---")
        report = generate_evaluation_report(
            agent_id="agent-1",
            metrics=current_metrics,
            trajectory_results=[result],
            drift_report=drift,
            ab_results=[ab_result],
        )
        print(f"Report for: {report.agent_id}")
        print(f"  Timestamp: {report.report_timestamp}")
        print("  Recommendations:")
        for rec in report.recommendations:
            print(f"    - {rec}")

        # 6. Accuracy evaluation
        print("\n--- Response Accuracy ---")
        acc_result = evaluate_response_accuracy(
            "The capital of France is Paris.",
            "The capital of france is paris.",
        )
        print(f"Accuracy: {acc_result.score:.2f} ({acc_result.details})")

        # 7. Precision/Recall
        print("\n--- Precision/Recall ---")
        precision, recall, f1 = calculate_precision_recall(
            ["search", "parse", "format", "extra"],
            ["search", "parse", "format", "validate"],
        )
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

        print("\n" + "=" * 60)
        print("Demo complete!")

    asyncio.run(demo())
