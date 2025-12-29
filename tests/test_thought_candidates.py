"""Tests for the Thought Candidates (Best-of-N) Pattern implementation."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns.thought_candidates import BestOfNResult
from agentic_patterns.thought_candidates import EvaluationContext
from agentic_patterns.thought_candidates import GenerationContext
from agentic_patterns.thought_candidates import ProblemStatement
from agentic_patterns.thought_candidates import ScoredThought
from agentic_patterns.thought_candidates import Thought
from agentic_patterns.thought_candidates import ThoughtEvaluation
from agentic_patterns.thought_candidates import evaluate_thought
from agentic_patterns.thought_candidates import generate_and_evaluate
from agentic_patterns.thought_candidates import generate_thought
from agentic_patterns.thought_candidates import run_best_of_n


class TestModels:
    """Test Pydantic model validation."""

    def test_problem_statement_minimal(self):
        problem = ProblemStatement(description="Solve for x")
        assert problem.description == "Solve for x"
        assert problem.constraints == []
        assert problem.context is None

    def test_problem_statement_full(self):
        problem = ProblemStatement(
            description="Game of 24",
            constraints=["Use each number once", "Result equals 24"],
            context="Previous step was 8 * 3 = 24",
        )
        assert len(problem.constraints) == 2
        assert problem.context is not None

    def test_thought_valid(self):
        thought = Thought(
            content="Multiply 8 by 3 to get 24",
            reasoning="This uses two of the numbers efficiently",
        )
        assert thought.content == "Multiply 8 by 3 to get 24"
        assert "efficiently" in thought.reasoning

    def test_thought_evaluation_valid(self):
        evaluation = ThoughtEvaluation(
            score=8.5,
            is_valid=True,
            feedback="Good approach, mathematically correct",
        )
        assert evaluation.score == 8.5
        assert evaluation.is_valid is True

    def test_thought_evaluation_score_bounds(self):
        # Valid at boundaries
        ThoughtEvaluation(score=0.0, is_valid=False, feedback="Terrible")
        ThoughtEvaluation(score=10.0, is_valid=True, feedback="Perfect")

        # Invalid scores
        with pytest.raises(ValueError):
            ThoughtEvaluation(score=-1.0, is_valid=False, feedback="Invalid")

        with pytest.raises(ValueError):
            ThoughtEvaluation(score=11.0, is_valid=True, feedback="Invalid")

    def test_scored_thought_computed_score(self):
        thought = Thought(content="Test", reasoning="Test reasoning")
        evaluation = ThoughtEvaluation(
            score=7.5, is_valid=True, feedback="Good"
        )
        scored = ScoredThought(thought=thought, evaluation=evaluation)

        # computed_field should expose the score
        assert scored.score == 7.5

    def test_best_of_n_result_valid(self):
        problem = ProblemStatement(description="Test problem")
        thought = Thought(content="Solution", reasoning="Reason")
        evaluation = ThoughtEvaluation(
            score=9.0, is_valid=True, feedback="Great"
        )
        scored = ScoredThought(thought=thought, evaluation=evaluation)

        result = BestOfNResult(
            problem=problem,
            candidates=[scored],
            best=scored,
            generation_count=1,
        )
        assert result.generation_count == 1
        assert result.best.score == 9.0

    def test_generation_context_has_problem(self):
        problem = ProblemStatement(description="Test")
        ctx = GenerationContext(problem=problem)
        assert ctx.problem.description == "Test"

    def test_evaluation_context_has_problem_and_thought(self):
        problem = ProblemStatement(description="Test")
        thought = Thought(content="Approach", reasoning="Because")
        ctx = EvaluationContext(problem=problem, thought=thought)
        assert ctx.problem.description == "Test"
        assert ctx.thought.content == "Approach"


class TestThoughtGeneration:
    """Test thought generation and evaluation functions."""

    @pytest.fixture
    def sample_problem(self):
        return ProblemStatement(
            description="Use 4, 7, 8, 8 to make 24",
            constraints=["Use each number once"],
        )

    @pytest.fixture
    def sample_thought(self):
        return Thought(
            content="8 / (1 - 7/8) = 8 / (1/8) = 64... wait, that's wrong",
            reasoning="Try division approach",
        )

    @pytest.fixture
    def sample_evaluation(self):
        return ThoughtEvaluation(
            score=6.5,
            is_valid=False,
            feedback="Math error in the calculation",
        )

    @pytest.mark.asyncio
    async def test_generate_thought_returns_thought(self, sample_problem):
        """Test that generate_thought returns a Thought object."""
        mock_thought = Thought(
            content="Try (8 - 4) * (8 - 7) = 4",
            reasoning="Subtract to get small numbers",
        )

        async def mock_run(prompt, deps):
            result = MagicMock()
            result.output = mock_thought
            return result

        with patch(
            "agentic_patterns.thought_candidates.generator_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(side_effect=mock_run)

            thought = await generate_thought(sample_problem)

            assert isinstance(thought, Thought)
            assert thought.content == mock_thought.content
            mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_thought_passes_context(self, sample_problem):
        """Test that context is passed through deps."""
        captured_deps = None

        async def mock_run(prompt, deps):
            nonlocal captured_deps
            captured_deps = deps
            result = MagicMock()
            result.output = Thought(content="Test", reasoning="Test")
            return result

        with patch(
            "agentic_patterns.thought_candidates.generator_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(side_effect=mock_run)

            await generate_thought(sample_problem)

            assert captured_deps is not None
            assert isinstance(captured_deps, GenerationContext)
            assert captured_deps.problem == sample_problem

    @pytest.mark.asyncio
    async def test_evaluate_thought_returns_evaluation(
        self, sample_problem, sample_thought, sample_evaluation
    ):
        """Test that evaluate_thought returns a ThoughtEvaluation."""

        async def mock_run(prompt, deps):
            result = MagicMock()
            result.output = sample_evaluation
            return result

        with patch(
            "agentic_patterns.thought_candidates.evaluator_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(side_effect=mock_run)

            evaluation = await evaluate_thought(sample_problem, sample_thought)

            assert isinstance(evaluation, ThoughtEvaluation)
            assert evaluation.score == 6.5
            mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_thought_passes_context(
        self, sample_problem, sample_thought
    ):
        """Test that problem and thought are passed through deps."""
        captured_deps = None

        async def mock_run(prompt, deps):
            nonlocal captured_deps
            captured_deps = deps
            result = MagicMock()
            result.output = ThoughtEvaluation(
                score=5.0, is_valid=True, feedback="OK"
            )
            return result

        with patch(
            "agentic_patterns.thought_candidates.evaluator_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(side_effect=mock_run)

            await evaluate_thought(sample_problem, sample_thought)

            assert captured_deps is not None
            assert isinstance(captured_deps, EvaluationContext)
            assert captured_deps.problem == sample_problem
            assert captured_deps.thought == sample_thought

    @pytest.mark.asyncio
    async def test_generate_and_evaluate_returns_scored_thought(
        self, sample_problem
    ):
        """Test atomic generate + evaluate operation."""
        mock_thought = Thought(content="Solution", reasoning="Reason")
        mock_evaluation = ThoughtEvaluation(
            score=8.0, is_valid=True, feedback="Good"
        )

        async def mock_gen_run(prompt, deps):
            result = MagicMock()
            result.output = mock_thought
            return result

        async def mock_eval_run(prompt, deps):
            result = MagicMock()
            result.output = mock_evaluation
            return result

        with (
            patch(
                "agentic_patterns.thought_candidates.generator_agent"
            ) as mock_gen,
            patch(
                "agentic_patterns.thought_candidates.evaluator_agent"
            ) as mock_eval,
        ):
            mock_gen.run = AsyncMock(side_effect=mock_gen_run)
            mock_eval.run = AsyncMock(side_effect=mock_eval_run)

            scored = await generate_and_evaluate(sample_problem)

            assert isinstance(scored, ScoredThought)
            assert scored.thought == mock_thought
            assert scored.evaluation == mock_evaluation
            assert scored.score == 8.0


class TestBestOfN:
    """Test the best-of-N sampling pattern."""

    @pytest.fixture
    def sample_problem(self):
        return ProblemStatement(
            description="Game of 24: Use 4, 7, 8, 8",
            constraints=["Result must equal 24"],
        )

    @pytest.fixture
    def mock_scored_thoughts(self):
        """Create mock scored thoughts with varying scores."""
        return [
            ScoredThought(
                thought=Thought(content="Approach A", reasoning="R1"),
                evaluation=ThoughtEvaluation(
                    score=6.0, is_valid=True, feedback="OK"
                ),
            ),
            ScoredThought(
                thought=Thought(content="Approach B", reasoning="R2"),
                evaluation=ThoughtEvaluation(
                    score=9.0, is_valid=True, feedback="Excellent"
                ),
            ),
            ScoredThought(
                thought=Thought(content="Approach C", reasoning="R3"),
                evaluation=ThoughtEvaluation(
                    score=7.5, is_valid=True, feedback="Good"
                ),
            ),
        ]

    @pytest.mark.asyncio
    async def test_run_best_of_n_generates_n_candidates(
        self, sample_problem, mock_scored_thoughts
    ):
        """Test that N candidates are generated."""
        thoughts_iter = iter(mock_scored_thoughts)

        async def mock_gen_eval(problem, config=None):
            return next(thoughts_iter)

        with patch(
            "agentic_patterns.thought_candidates.generate_and_evaluate",
            side_effect=mock_gen_eval,
        ):
            result = await run_best_of_n(sample_problem, n=3)

            assert result.generation_count == 3
            assert len(result.candidates) == 3

    @pytest.mark.asyncio
    async def test_run_best_of_n_selects_highest_score(
        self, sample_problem, mock_scored_thoughts
    ):
        """Test that the best candidate is the highest-scoring one."""
        thoughts_iter = iter(mock_scored_thoughts)

        async def mock_gen_eval(problem, config=None):
            return next(thoughts_iter)

        with patch(
            "agentic_patterns.thought_candidates.generate_and_evaluate",
            side_effect=mock_gen_eval,
        ):
            result = await run_best_of_n(sample_problem, n=3)

            # Score 9.0 should be best
            assert result.best.score == 9.0
            assert result.best.thought.content == "Approach B"

    @pytest.mark.asyncio
    async def test_run_best_of_n_sorts_candidates_descending(
        self, sample_problem, mock_scored_thoughts
    ):
        """Test that candidates are sorted by score descending."""
        thoughts_iter = iter(mock_scored_thoughts)

        async def mock_gen_eval(problem, config=None):
            return next(thoughts_iter)

        with patch(
            "agentic_patterns.thought_candidates.generate_and_evaluate",
            side_effect=mock_gen_eval,
        ):
            result = await run_best_of_n(sample_problem, n=3)

            scores = [c.score for c in result.candidates]
            assert scores == sorted(scores, reverse=True)
            assert scores == [9.0, 7.5, 6.0]

    @pytest.mark.asyncio
    async def test_run_best_of_n_returns_problem(self, sample_problem):
        """Test that the problem is included in the result."""
        mock_scored = ScoredThought(
            thought=Thought(content="X", reasoning="Y"),
            evaluation=ThoughtEvaluation(
                score=5.0, is_valid=True, feedback="Z"
            ),
        )

        async def mock_gen_eval(problem, config=None):
            return mock_scored

        with patch(
            "agentic_patterns.thought_candidates.generate_and_evaluate",
            side_effect=mock_gen_eval,
        ):
            result = await run_best_of_n(sample_problem, n=1)

            assert result.problem == sample_problem
            assert result.problem.description == "Game of 24: Use 4, 7, 8, 8"

    @pytest.mark.asyncio
    async def test_run_best_of_n_parallel_execution(self, sample_problem):
        """Test that candidates are generated in parallel."""
        call_times = []
        import time

        mock_scored = ScoredThought(
            thought=Thought(content="X", reasoning="Y"),
            evaluation=ThoughtEvaluation(
                score=5.0, is_valid=True, feedback="Z"
            ),
        )

        async def mock_gen_eval(problem, config=None):
            call_times.append(time.time())
            return mock_scored

        with patch(
            "agentic_patterns.thought_candidates.generate_and_evaluate",
            side_effect=mock_gen_eval,
        ):
            await run_best_of_n(sample_problem, n=5)

            # All calls should happen nearly simultaneously (parallel)
            assert len(call_times) == 5
            # The spread should be very small if truly parallel
            time_spread = max(call_times) - min(call_times)
            assert time_spread < 0.1  # All within 100ms


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_problem_statement_empty_constraints(self):
        problem = ProblemStatement(description="Simple problem")
        assert problem.constraints == []

    def test_thought_evaluation_boundary_scores(self):
        # Minimum valid score
        eval_min = ThoughtEvaluation(
            score=0.0, is_valid=False, feedback="Completely wrong"
        )
        assert eval_min.score == 0.0

        # Maximum valid score
        eval_max = ThoughtEvaluation(
            score=10.0, is_valid=True, feedback="Perfect solution"
        )
        assert eval_max.score == 10.0

    def test_scored_thought_immutable_score(self):
        """Score is computed from evaluation, can't be directly set."""
        thought = Thought(content="X", reasoning="Y")
        evaluation = ThoughtEvaluation(
            score=7.0, is_valid=True, feedback="Good"
        )
        scored = ScoredThought(thought=thought, evaluation=evaluation)

        # The score property comes from evaluation
        assert scored.score == 7.0

    def test_best_of_n_result_single_candidate(self):
        """Test with just one candidate."""
        problem = ProblemStatement(description="Test")
        thought = Thought(content="Only option", reasoning="No choice")
        evaluation = ThoughtEvaluation(score=5.0, is_valid=True, feedback="OK")
        scored = ScoredThought(thought=thought, evaluation=evaluation)

        result = BestOfNResult(
            problem=problem,
            candidates=[scored],
            best=scored,
            generation_count=1,
        )
        assert result.best == result.candidates[0]
        assert len(result.candidates) == 1

    @pytest.mark.asyncio
    async def test_run_best_of_n_with_n_equals_1(self):
        """Test best-of-N with N=1 (degenerates to single generation)."""
        problem = ProblemStatement(description="Test")
        mock_scored = ScoredThought(
            thought=Thought(content="Single", reasoning="Only one"),
            evaluation=ThoughtEvaluation(
                score=8.0, is_valid=True, feedback="OK"
            ),
        )

        async def mock_gen_eval(p, config=None):
            return mock_scored

        with patch(
            "agentic_patterns.thought_candidates.generate_and_evaluate",
            side_effect=mock_gen_eval,
        ):
            result = await run_best_of_n(problem, n=1)

            assert result.generation_count == 1
            assert len(result.candidates) == 1
            assert result.best == result.candidates[0]

    def test_problem_with_context_chaining(self):
        """Test that context field supports chaining from previous steps."""
        problem = ProblemStatement(
            description="Continue solving",
            context="Previous step: 8 * 3 = 24",
        )
        assert problem.context is not None
        assert "24" in problem.context
