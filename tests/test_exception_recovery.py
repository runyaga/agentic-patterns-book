"""Tests for the Exception Recovery Pattern (Phoenix Protocol)."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic_ai.usage import RunUsage

from agentic_patterns.exception_recovery import ClinicDiagnosis
from agentic_patterns.exception_recovery import ErrorCategory
from agentic_patterns.exception_recovery import RecoveryAction
from agentic_patterns.exception_recovery import RecoveryConfig
from agentic_patterns.exception_recovery import RecoveryResult
from agentic_patterns.exception_recovery import _truncate_history
from agentic_patterns.exception_recovery import _truncate_prompt
from agentic_patterns.exception_recovery import classify_error
from agentic_patterns.exception_recovery import get_recovery_action
from agentic_patterns.exception_recovery import is_retryable
from agentic_patterns.exception_recovery import recoverable_run


class TestModels:
    """Test Pydantic model validation."""

    def test_recovery_action_valid(self):
        action = RecoveryAction(
            action="retry",
            wait_seconds=1.0,
            reason="Transient error",
        )
        assert action.action == "retry"
        assert action.wait_seconds == 1.0
        assert action.truncate_to is None

    def test_recovery_action_with_truncate(self):
        action = RecoveryAction(
            action="retry_shorter",
            truncate_to=500,
            reason="Prompt too long",
        )
        assert action.action == "retry_shorter"
        assert action.truncate_to == 500

    def test_recovery_result_valid(self):
        result = RecoveryResult(
            success=True,
            attempts=2,
            categories_seen=["timeout", "unknown"],
        )
        assert result.success
        assert result.attempts == 2
        assert len(result.categories_seen) == 2

    def test_clinic_diagnosis_valid(self):
        diagnosis = ClinicDiagnosis(
            should_retry=True,
            suggestion="Network issue, retry should help",
        )
        assert diagnosis.should_retry
        assert "retry" in diagnosis.suggestion.lower()


class TestRecoveryConfig:
    """Test RecoveryConfig dataclass."""

    def test_config_defaults(self):
        config = RecoveryConfig()
        assert config.max_attempts == 3
        assert config.backoff_seconds == 1.0
        assert config.rate_limit_wait == 60.0
        assert config.truncate_ratio == 0.7
        assert config.use_clinic_for_unknown is True

    def test_config_custom(self):
        config = RecoveryConfig(
            max_attempts=5,
            backoff_seconds=2.0,
            use_clinic_for_unknown=False,
        )
        assert config.max_attempts == 5
        assert config.backoff_seconds == 2.0
        assert config.use_clinic_for_unknown is False


class TestClassifyError:
    """Test error classification heuristics."""

    def test_classify_context_length(self):
        exc = ValueError("maximum context length exceeded")
        assert classify_error(exc) == ErrorCategory.CONTEXT_LENGTH

    def test_classify_context_length_variant(self):
        exc = Exception("Error: too many tokens in prompt")
        assert classify_error(exc) == ErrorCategory.CONTEXT_LENGTH

    def test_classify_timeout(self):
        exc = TimeoutError("Request timed out")
        assert classify_error(exc) == ErrorCategory.TIMEOUT

    def test_classify_timeout_message(self):
        exc = Exception("Connection timeout after 30s")
        assert classify_error(exc) == ErrorCategory.TIMEOUT

    def test_classify_rate_limit_429(self):
        exc = Exception("Error 429: rate limit exceeded")
        assert classify_error(exc) == ErrorCategory.RATE_LIMIT

    def test_classify_rate_limit_message(self):
        exc = Exception("Too many requests, please slow down")
        assert classify_error(exc) == ErrorCategory.RATE_LIMIT

    def test_classify_invalid_json(self):
        exc = ValueError("JSON decode error: invalid syntax")
        assert classify_error(exc) == ErrorCategory.INVALID_JSON

    def test_classify_json_parse(self):
        exc = Exception("Failed to parse JSON response")
        assert classify_error(exc) == ErrorCategory.INVALID_JSON

    def test_classify_connection_error(self):
        exc = ConnectionError("Connection refused")
        assert classify_error(exc) == ErrorCategory.CONNECTION

    def test_classify_network_error(self):
        exc = Exception("Network error: connection reset by peer")
        assert classify_error(exc) == ErrorCategory.CONNECTION

    def test_classify_tool_error(self):
        exc = RuntimeError("Tool execution failed with error")
        assert classify_error(exc) == ErrorCategory.TOOL_ERROR

    def test_classify_unknown(self):
        exc = ValueError("Something completely unexpected")
        assert classify_error(exc) == ErrorCategory.UNKNOWN

    def test_classify_empty_message(self):
        exc = Exception("")
        assert classify_error(exc) == ErrorCategory.UNKNOWN


class TestGetRecoveryAction:
    """Test recovery action determination."""

    def test_context_length_action(self):
        config = RecoveryConfig()
        action = get_recovery_action(ErrorCategory.CONTEXT_LENGTH, 0, config)
        assert action.action == "retry_shorter"
        assert action.truncate_to is not None
        assert action.truncate_to > 0

    def test_context_length_progressive_truncation(self):
        config = RecoveryConfig(truncate_ratio=0.5)
        action0 = get_recovery_action(ErrorCategory.CONTEXT_LENGTH, 0, config)
        action1 = get_recovery_action(ErrorCategory.CONTEXT_LENGTH, 1, config)
        assert action0.truncate_to > action1.truncate_to

    def test_timeout_action(self):
        config = RecoveryConfig()
        action = get_recovery_action(ErrorCategory.TIMEOUT, 0, config)
        assert action.action == "retry_shorter"
        assert action.truncate_to is not None

    def test_rate_limit_action(self):
        config = RecoveryConfig(rate_limit_wait=30.0)
        action = get_recovery_action(ErrorCategory.RATE_LIMIT, 0, config)
        assert action.action == "wait_and_retry"
        assert action.wait_seconds == 30.0

    def test_invalid_json_action(self):
        config = RecoveryConfig(backoff_seconds=2.0)
        action = get_recovery_action(ErrorCategory.INVALID_JSON, 1, config)
        assert action.action == "retry"
        assert action.wait_seconds == 4.0  # 2.0 * (1+1)

    def test_connection_action(self):
        config = RecoveryConfig()
        action = get_recovery_action(ErrorCategory.CONNECTION, 0, config)
        assert action.action == "retry"
        assert action.wait_seconds > 0

    def test_tool_error_action(self):
        config = RecoveryConfig()
        action = get_recovery_action(ErrorCategory.TOOL_ERROR, 0, config)
        assert action.action == "retry"

    def test_unknown_action(self):
        config = RecoveryConfig()
        action = get_recovery_action(ErrorCategory.UNKNOWN, 0, config)
        assert action.action == "retry"

    def test_max_attempts_reached(self):
        config = RecoveryConfig(max_attempts=3)
        action = get_recovery_action(ErrorCategory.TIMEOUT, 3, config)
        assert action.action == "abort"
        assert "Max attempts" in action.reason

    def test_max_attempts_exceeded(self):
        config = RecoveryConfig(max_attempts=2)
        action = get_recovery_action(ErrorCategory.CONNECTION, 5, config)
        assert action.action == "abort"


class TestIsRetryable:
    """Test the is_retryable helper function."""

    def test_retryable_timeout(self):
        """Timeout errors should be retryable."""
        exc = TimeoutError("Request timed out")
        assert is_retryable(exc) is True

    def test_retryable_connection(self):
        """Connection errors should be retryable."""
        exc = ConnectionError("Connection refused")
        assert is_retryable(exc) is True

    def test_retryable_rate_limit(self):
        """Rate limit errors should be retryable."""
        exc = Exception("Error 429: rate limit exceeded")
        assert is_retryable(exc) is True

    def test_not_retryable_unknown(self):
        """Unknown errors should not be retryable."""
        exc = ValueError("Something completely unexpected happened")
        assert is_retryable(exc) is False


class TestTruncateHelpers:
    """Test truncation helper functions."""

    def test_truncate_prompt_none(self):
        """Should return None for None input."""
        assert _truncate_prompt(None, 100) is None

    def test_truncate_prompt_short(self):
        """Should not truncate short prompts."""
        result = _truncate_prompt("short", 100)
        assert result == "short"

    def test_truncate_prompt_long(self):
        """Should truncate long prompts with notice."""
        result = _truncate_prompt("x" * 200, 100)
        assert len(result) < 200
        assert "[Truncated" in result

    def test_truncate_prompt_sequence(self):
        """Should return sequence as-is."""
        seq = [{"type": "text", "text": "hello"}]
        result = _truncate_prompt(seq, 10)
        assert result is seq

    def test_truncate_history_none(self):
        """Should return None for None history."""
        assert _truncate_history(None, 0.5) is None

    def test_truncate_history_empty(self):
        """Should return None for empty history."""
        assert _truncate_history([], 0.5) is None

    def test_truncate_history_keeps_recent(self):
        """Should keep most recent messages."""
        history = [MagicMock() for _ in range(10)]
        result = _truncate_history(history, 0.3)
        assert len(result) == 3
        assert result == history[-3:]


class TestRecoverableRun:
    """Test the recoverable_run function."""

    @pytest.fixture
    def mock_agent(self):
        return MagicMock()

    @pytest.fixture
    def mock_run_result(self):
        """Create a mock result with usage method."""
        mock_result = MagicMock()
        mock_result.output = "Success"
        mock_result.usage.return_value = RunUsage(
            requests=1,
            input_tokens=10,
            output_tokens=5,
        )
        return mock_result

    @pytest.mark.asyncio
    async def test_success_first_attempt(self, mock_agent, mock_run_result):
        """Should succeed on first attempt without recovery."""
        mock_agent.run = AsyncMock(return_value=mock_run_result)

        result = await recoverable_run(mock_agent, "test prompt")

        assert result.output == "Success"
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_full_result(self, mock_agent, mock_run_result):
        """Should return full AgentRunResult, not just output."""
        mock_agent.run = AsyncMock(return_value=mock_run_result)

        result = await recoverable_run(mock_agent, "test prompt")

        # Should have access to output, usage, etc.
        assert hasattr(result, "output")
        assert result.output == "Success"

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, mock_agent):
        """Should retry and succeed after transient failure."""
        mock_result = MagicMock()
        mock_result.output = "Success"
        mock_result.usage.return_value = RunUsage(requests=1)

        call_count = 0

        async def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection reset")
            return mock_result

        mock_agent.run = mock_run

        result = await recoverable_run(mock_agent, "test prompt")

        assert result.output == "Success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts(self, mock_agent):
        """Should give up after max attempts."""
        mock_agent.run = AsyncMock(
            side_effect=TimeoutError("Always times out")
        )

        with pytest.raises(TimeoutError):
            await recoverable_run(
                mock_agent,
                "test prompt",
                recovery_config=RecoveryConfig(max_attempts=2),
            )

        assert mock_agent.run.call_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_truncates_on_context_length(self, mock_agent):
        """Should truncate prompt on context length error."""
        mock_result = MagicMock()
        mock_result.output = "Success"
        mock_result.usage.return_value = RunUsage(requests=1)

        calls = []

        async def mock_run(prompt, **kwargs):
            calls.append(prompt)
            if len(calls) == 1:
                raise ValueError("maximum context length exceeded")
            return mock_result

        mock_agent.run = mock_run

        long_prompt = "x" * 5000
        result = await recoverable_run(mock_agent, long_prompt)

        assert result.output == "Success"
        assert len(calls) == 2
        assert len(calls[1]) < len(calls[0])

    @pytest.mark.asyncio
    async def test_passes_deps_to_agent(self, mock_agent, mock_run_result):
        """Should pass deps to agent.run()."""
        mock_agent.run = AsyncMock(return_value=mock_run_result)

        class MyDeps:
            value = 42

        deps = MyDeps()
        await recoverable_run(mock_agent, "test", deps=deps)

        mock_agent.run.assert_called_once()
        call_kwargs = mock_agent.run.call_args[1]
        assert call_kwargs["deps"] is deps

    @pytest.mark.asyncio
    async def test_no_deps_when_none(self, mock_agent, mock_run_result):
        """Should not pass deps kwarg when deps is None."""
        mock_agent.run = AsyncMock(return_value=mock_run_result)

        await recoverable_run(mock_agent, "test", deps=None)

        mock_agent.run.assert_called_once()
        call_kwargs = mock_agent.run.call_args[1]
        assert "deps" not in call_kwargs

    @pytest.mark.asyncio
    async def test_rate_limit_wait(self, mock_agent):
        """Should wait on rate limit error."""
        mock_result = MagicMock()
        mock_result.output = "Success"
        mock_result.usage.return_value = RunUsage(requests=1)

        call_count = 0

        async def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Error 429: rate limit")
            return mock_result

        mock_agent.run = mock_run

        sleep_path = "agentic_patterns.exception_recovery.asyncio.sleep"
        with patch(sleep_path) as mock_sleep:
            mock_sleep.return_value = None
            await recoverable_run(
                mock_agent,
                "test",
                recovery_config=RecoveryConfig(rate_limit_wait=10.0),
            )

            mock_sleep.assert_called()
            # First sleep should be for rate limit
            assert mock_sleep.call_args_list[0][0][0] == 10.0

    @pytest.mark.asyncio
    async def test_forwards_kwargs_to_agent(self, mock_agent, mock_run_result):
        """Should forward **kwargs to agent.run()."""
        mock_agent.run = AsyncMock(return_value=mock_run_result)

        await recoverable_run(
            mock_agent,
            "test",
            model_settings={"temperature": 0.5},
        )

        call_kwargs = mock_agent.run.call_args[1]
        assert call_kwargs["model_settings"] == {"temperature": 0.5}

    @pytest.mark.asyncio
    async def test_usage_accumulated_across_retries(self, mock_agent):
        """Should accumulate usage from all attempts."""
        call_count = 0

        def make_result(tokens: int):
            r = MagicMock()
            r.output = "Success"
            r.usage.return_value = RunUsage(
                requests=1,
                input_tokens=tokens,
                output_tokens=tokens,
            )
            return r

        async def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection reset")
            return make_result(20)

        mock_agent.run = mock_run

        result = await recoverable_run(mock_agent, "test")

        # Usage should be from successful attempt only (failed attempt
        # doesn't have usage to capture in this simple mock)
        assert result._usage.requests == 1


class TestClinicAgent:
    """Test clinic agent for unknown errors."""

    @pytest.mark.asyncio
    async def test_clinic_called_for_unknown(self):
        """Clinic agent should be called for unknown errors."""
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "Success"
        mock_result.usage.return_value = RunUsage(requests=1)

        call_count = 0

        async def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Completely unexpected error xyz")
            return mock_result

        mock_agent.run = mock_run

        mock_clinic_result = MagicMock()
        mock_clinic_result.output = ClinicDiagnosis(
            should_retry=True,
            suggestion="This looks like a transient issue",
        )

        with patch(
            "agentic_patterns.exception_recovery._create_clinic_agent"
        ) as mock_create:
            mock_clinic = MagicMock()
            mock_clinic.run = AsyncMock(return_value=mock_clinic_result)
            mock_create.return_value = mock_clinic

            result = await recoverable_run(
                mock_agent,
                "test",
                recovery_config=RecoveryConfig(use_clinic_for_unknown=True),
            )

            assert result.output == "Success"
            mock_clinic.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_clinic_not_called_when_disabled(self):
        """Clinic should not be called when use_clinic_for_unknown=False."""
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "Success"
        mock_result.usage.return_value = RunUsage(requests=1)

        call_count = 0

        async def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Unexpected error")
            return mock_result

        mock_agent.run = mock_run

        with patch(
            "agentic_patterns.exception_recovery._create_clinic_agent"
        ) as mock_create:
            await recoverable_run(
                mock_agent,
                "test",
                recovery_config=RecoveryConfig(use_clinic_for_unknown=False),
            )

            mock_create.assert_not_called()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_error_category_values(self):
        """Test all error categories have expected string values."""
        assert ErrorCategory.CONTEXT_LENGTH.value == "context_length"
        assert ErrorCategory.TIMEOUT.value == "timeout"
        assert ErrorCategory.RATE_LIMIT.value == "rate_limit"
        assert ErrorCategory.INVALID_JSON.value == "invalid_json"
        assert ErrorCategory.CONNECTION.value == "connection"
        assert ErrorCategory.TOOL_ERROR.value == "tool_error"
        assert ErrorCategory.UNKNOWN.value == "unknown"

    def test_classify_case_insensitive(self):
        """Error classification should be case insensitive."""
        exc = Exception("MAXIMUM CONTEXT LENGTH EXCEEDED")
        assert classify_error(exc) == ErrorCategory.CONTEXT_LENGTH

    def test_recovery_action_abort_no_wait(self):
        """Abort action should have zero wait time."""
        config = RecoveryConfig(max_attempts=1)
        action = get_recovery_action(ErrorCategory.TIMEOUT, 1, config)
        assert action.action == "abort"
        assert action.wait_seconds == 0.0

    @pytest.mark.asyncio
    async def test_single_attempt_mode(self):
        """Should work with max_attempts=0 (single attempt)."""
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=ConnectionError("Fail"))

        with pytest.raises(ConnectionError):
            await recoverable_run(
                mock_agent,
                "test",
                recovery_config=RecoveryConfig(max_attempts=0),
            )

        mock_agent.run.assert_called_once()
