"""Tests for the Tree of Thoughts Pattern implementation."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns.thought_candidates import ProblemStatement
from agentic_patterns.thought_candidates import ScoredThought
from agentic_patterns.thought_candidates import Thought
from agentic_patterns.thought_candidates import ThoughtEvaluation
from agentic_patterns.tree_of_thoughts import SynthesisContext
from agentic_patterns.tree_of_thoughts import SynthesizedSolution
from agentic_patterns.tree_of_thoughts import ThoughtNode
from agentic_patterns.tree_of_thoughts import TreeConfig
from agentic_patterns.tree_of_thoughts import TreeExplorationResult
from agentic_patterns.tree_of_thoughts import beam_search
from agentic_patterns.tree_of_thoughts import expand_node
from agentic_patterns.tree_of_thoughts import run_tree_of_thoughts
from agentic_patterns.tree_of_thoughts import synthesize_solution
from agentic_patterns.tree_of_thoughts import trace_best_path


class TestModels:
    """Test Pydantic model validation."""

    @pytest.fixture
    def sample_scored_thought(self):
        return ScoredThought(
            thought=Thought(content="Test approach", reasoning="Test reason"),
            evaluation=ThoughtEvaluation(
                score=7.5, is_valid=True, feedback="Good"
            ),
        )

    def test_thought_node_valid(self, sample_scored_thought):
        node = ThoughtNode(
            id="0.1",
            depth=1,
            scored_thought=sample_scored_thought,
            parent_id="0.0",
        )
        assert node.id == "0.1"
        assert node.depth == 1
        assert node.parent_id == "0.0"
        assert node.children_ids == []
        assert node.is_pruned is False

    def test_thought_node_computed_score(self, sample_scored_thought):
        node = ThoughtNode(
            id="0.0",
            depth=0,
            scored_thought=sample_scored_thought,
        )
        assert node.score == 7.5

    def test_thought_node_with_children(self, sample_scored_thought):
        node = ThoughtNode(
            id="0.0",
            depth=0,
            scored_thought=sample_scored_thought,
            children_ids=["0.0.0", "0.0.1", "0.0.2"],
        )
        assert len(node.children_ids) == 3

    def test_thought_node_pruned(self, sample_scored_thought):
        node = ThoughtNode(
            id="0.0",
            depth=0,
            scored_thought=sample_scored_thought,
            is_pruned=True,
        )
        assert node.is_pruned is True

    def test_tree_config_defaults(self):
        config = TreeConfig()
        assert config.max_depth == 3
        assert config.branch_factor == 3
        assert config.prune_threshold == 5.0
        assert config.beam_width == 3

    def test_tree_config_custom(self):
        config = TreeConfig(
            max_depth=5,
            branch_factor=4,
            prune_threshold=6.0,
            beam_width=2,
        )
        assert config.max_depth == 5
        assert config.branch_factor == 4
        assert config.prune_threshold == 6.0
        assert config.beam_width == 2

    def test_tree_config_bounds(self):
        # Valid at boundaries
        TreeConfig(max_depth=1)
        TreeConfig(max_depth=10)
        TreeConfig(prune_threshold=0.0)
        TreeConfig(prune_threshold=10.0)

        # Invalid
        with pytest.raises(ValueError):
            TreeConfig(max_depth=0)

        with pytest.raises(ValueError):
            TreeConfig(max_depth=11)

        with pytest.raises(ValueError):
            TreeConfig(prune_threshold=-1.0)

    def test_synthesized_solution_valid(self):
        solution = SynthesizedSolution(
            solution="8 * (8 - 4 + 7/7) = 24",
            confidence=0.95,
            reasoning="Combined operations to reach 24",
        )
        assert solution.solution == "8 * (8 - 4 + 7/7) = 24"
        assert solution.confidence == 0.95

    def test_synthesized_solution_confidence_bounds(self):
        SynthesizedSolution(
            solution="X", confidence=0.0, reasoning="Zero confidence"
        )
        SynthesizedSolution(
            solution="X", confidence=1.0, reasoning="Full confidence"
        )

        with pytest.raises(ValueError):
            SynthesizedSolution(
                solution="X", confidence=1.5, reasoning="Invalid"
            )

    def test_tree_exploration_result_valid(self, sample_scored_thought):
        problem = ProblemStatement(description="Test problem")
        config = TreeConfig()
        node = ThoughtNode(
            id="0.0", depth=0, scored_thought=sample_scored_thought
        )
        solution = SynthesizedSolution(
            solution="Answer", confidence=0.9, reasoning="Derived"
        )

        result = TreeExplorationResult(
            problem=problem,
            config=config,
            all_nodes=[node],
            best_path=[node],
            solution=solution,
            nodes_explored=1,
            nodes_pruned=0,
        )
        assert result.nodes_explored == 1
        assert len(result.best_path) == 1

    def test_synthesis_context_fields(self, sample_scored_thought):
        problem = ProblemStatement(description="Test")
        node = ThoughtNode(
            id="0.0", depth=0, scored_thought=sample_scored_thought
        )
        ctx = SynthesisContext(problem=problem, path=[node])
        assert ctx.problem.description == "Test"
        assert len(ctx.path) == 1


class TestExpandNode:
    """Test node expansion functionality."""

    @pytest.fixture
    def sample_problem(self):
        return ProblemStatement(
            description="Game of 24",
            constraints=["Use each number once"],
        )

    @pytest.fixture
    def sample_parent_node(self):
        return ThoughtNode(
            id="0.0",
            depth=0,
            scored_thought=ScoredThought(
                thought=Thought(content="Start with 8*3", reasoning="R"),
                evaluation=ThoughtEvaluation(
                    score=7.0, is_valid=True, feedback="OK"
                ),
            ),
        )

    @pytest.fixture
    def sample_config(self):
        return TreeConfig(
            max_depth=3, branch_factor=3, prune_threshold=5.0, beam_width=2
        )

    @pytest.mark.asyncio
    async def test_expand_node_creates_children(
        self, sample_problem, sample_parent_node, sample_config
    ):
        """Test that expand_node creates the right number of children."""
        mock_scored = ScoredThought(
            thought=Thought(content="Child", reasoning="R"),
            evaluation=ThoughtEvaluation(
                score=6.0, is_valid=True, feedback="OK"
            ),
        )

        async def mock_gen_eval(problem, config=None, **kwargs):
            return mock_scored

        with patch(
            "agentic_patterns.tree_of_thoughts.generate_and_evaluate",
            side_effect=mock_gen_eval,
        ):
            children = await expand_node(
                sample_problem, sample_parent_node, sample_config
            )

            assert len(children) == sample_config.branch_factor
            assert len(sample_parent_node.children_ids) == 3

    @pytest.mark.asyncio
    async def test_expand_node_sets_correct_ids(
        self, sample_problem, sample_parent_node, sample_config
    ):
        """Test that child IDs are correctly formatted."""
        mock_scored = ScoredThought(
            thought=Thought(content="Child", reasoning="R"),
            evaluation=ThoughtEvaluation(
                score=6.0, is_valid=True, feedback="OK"
            ),
        )

        async def mock_gen_eval(problem, config=None, **kwargs):
            return mock_scored

        with patch(
            "agentic_patterns.tree_of_thoughts.generate_and_evaluate",
            side_effect=mock_gen_eval,
        ):
            children = await expand_node(
                sample_problem, sample_parent_node, sample_config
            )

            assert children[0].id == "0.0.0"
            assert children[1].id == "0.0.1"
            assert children[2].id == "0.0.2"
            assert all(c.parent_id == "0.0" for c in children)

    @pytest.mark.asyncio
    async def test_expand_node_increments_depth(
        self, sample_problem, sample_parent_node, sample_config
    ):
        """Test that child depth is parent depth + 1."""
        mock_scored = ScoredThought(
            thought=Thought(content="Child", reasoning="R"),
            evaluation=ThoughtEvaluation(
                score=6.0, is_valid=True, feedback="OK"
            ),
        )

        async def mock_gen_eval(problem, config=None, **kwargs):
            return mock_scored

        with patch(
            "agentic_patterns.tree_of_thoughts.generate_and_evaluate",
            side_effect=mock_gen_eval,
        ):
            children = await expand_node(
                sample_problem, sample_parent_node, sample_config
            )

            assert all(c.depth == 1 for c in children)

    @pytest.mark.asyncio
    async def test_expand_node_prunes_low_scores(
        self, sample_problem, sample_parent_node, sample_config
    ):
        """Test that nodes below threshold are marked as pruned."""
        scores = [6.0, 4.0, 3.0]  # Only first is above threshold (5.0)
        score_iter = iter(scores)

        async def mock_gen_eval(problem, config=None, **kwargs):
            score = next(score_iter)
            return ScoredThought(
                thought=Thought(content="Child", reasoning="R"),
                evaluation=ThoughtEvaluation(
                    score=score, is_valid=True, feedback="OK"
                ),
            )

        with patch(
            "agentic_patterns.tree_of_thoughts.generate_and_evaluate",
            side_effect=mock_gen_eval,
        ):
            children = await expand_node(
                sample_problem, sample_parent_node, sample_config
            )

            # Only first (6.0) is above threshold (5.0)
            assert children[0].is_pruned is False
            assert children[1].is_pruned is True
            assert children[2].is_pruned is True

    @pytest.mark.asyncio
    async def test_expand_node_respects_max_depth(
        self, sample_problem, sample_config
    ):
        """Test that nodes at max depth are not expanded."""
        deep_node = ThoughtNode(
            id="0.0.0",
            depth=3,  # At max_depth
            scored_thought=ScoredThought(
                thought=Thought(content="Deep", reasoning="R"),
                evaluation=ThoughtEvaluation(
                    score=8.0, is_valid=True, feedback="OK"
                ),
            ),
        )

        children = await expand_node(sample_problem, deep_node, sample_config)
        assert len(children) == 0

    @pytest.mark.asyncio
    async def test_expand_node_skips_pruned_nodes(
        self, sample_problem, sample_config
    ):
        """Test that pruned nodes are not expanded."""
        pruned_node = ThoughtNode(
            id="0.0",
            depth=0,
            scored_thought=ScoredThought(
                thought=Thought(content="Pruned", reasoning="R"),
                evaluation=ThoughtEvaluation(
                    score=3.0, is_valid=False, feedback="Bad"
                ),
            ),
            is_pruned=True,
        )

        children = await expand_node(
            sample_problem, pruned_node, sample_config
        )
        assert len(children) == 0


class TestTraceBestPath:
    """Test path tracing functionality."""

    def make_node(self, id, depth, score, parent_id=None, children=None):
        return ThoughtNode(
            id=id,
            depth=depth,
            scored_thought=ScoredThought(
                thought=Thought(content=f"Node {id}", reasoning="R"),
                evaluation=ThoughtEvaluation(
                    score=score, is_valid=True, feedback="OK"
                ),
            ),
            parent_id=parent_id,
            children_ids=children or [],
        )

    def test_trace_best_path_single_node(self):
        node = self.make_node("0.0", 0, 8.0)
        path = trace_best_path([node])
        assert len(path) == 1
        assert path[0].id == "0.0"

    def test_trace_best_path_linear_chain(self):
        nodes = [
            self.make_node("0.0", 0, 7.0, children=["0.0.0"]),
            self.make_node("0.0.0", 1, 8.0, "0.0", children=["0.0.0.0"]),
            self.make_node("0.0.0.0", 2, 9.0, "0.0.0"),
        ]

        path = trace_best_path(nodes)

        assert len(path) == 3
        assert [n.id for n in path] == ["0.0", "0.0.0", "0.0.0.0"]

    def test_trace_best_path_selects_highest_scoring_leaf(self):
        # Two branches, one with higher leaf score
        nodes = [
            self.make_node("0.0", 0, 7.0, children=["0.0.0", "0.0.1"]),
            self.make_node("0.0.0", 1, 6.0, "0.0"),  # Lower leaf
            self.make_node("0.0.1", 1, 9.0, "0.0"),  # Higher leaf
        ]

        path = trace_best_path(nodes)

        assert len(path) == 2
        assert path[0].id == "0.0"
        assert path[1].id == "0.0.1"  # Higher scoring leaf

    def test_trace_best_path_empty_nodes(self):
        path = trace_best_path([])
        assert path == []

    def test_trace_best_path_excludes_pruned_from_selection(self):
        nodes = [
            self.make_node("0.0", 0, 7.0, children=["0.0.0", "0.0.1"]),
            self.make_node("0.0.0", 1, 9.0, "0.0"),  # Higher but pruned
            self.make_node("0.0.1", 1, 6.0, "0.0"),  # Lower but not pruned
        ]
        nodes[1].is_pruned = True

        path = trace_best_path(nodes)

        # Should select the unpruned leaf
        assert path[-1].id == "0.0.1"


class TestBeamSearch:
    """Test beam search functionality."""

    @pytest.fixture
    def sample_problem(self):
        return ProblemStatement(description="Test problem")

    @pytest.fixture
    def sample_config(self):
        return TreeConfig(
            max_depth=2, branch_factor=2, prune_threshold=5.0, beam_width=2
        )

    @pytest.mark.asyncio
    async def test_beam_search_generates_root_nodes(
        self, sample_problem, sample_config
    ):
        """Test that beam search generates root nodes."""
        mock_scored = ScoredThought(
            thought=Thought(content="Root", reasoning="R"),
            evaluation=ThoughtEvaluation(
                score=7.0, is_valid=True, feedback="OK"
            ),
        )

        async def mock_gen_eval(problem, config=None, **kwargs):
            return mock_scored

        with patch(
            "agentic_patterns.tree_of_thoughts.generate_and_evaluate",
            side_effect=mock_gen_eval,
        ):
            all_nodes, best_path = await beam_search(
                sample_problem, sample_config
            )

            # Should have root nodes
            root_nodes = [n for n in all_nodes if n.depth == 0]
            assert len(root_nodes) == sample_config.branch_factor

    @pytest.mark.asyncio
    async def test_beam_search_respects_beam_width(
        self, sample_problem, sample_config
    ):
        """Test that only top-k nodes are expanded."""
        call_count = 0

        async def mock_gen_eval(problem, config=None, **kwargs):
            nonlocal call_count
            call_count += 1
            return ScoredThought(
                thought=Thought(content=f"Node {call_count}", reasoning="R"),
                evaluation=ThoughtEvaluation(
                    score=7.0, is_valid=True, feedback="OK"
                ),
            )

        with patch(
            "agentic_patterns.tree_of_thoughts.generate_and_evaluate",
            side_effect=mock_gen_eval,
        ):
            await beam_search(sample_problem, sample_config)

            # With max_depth=2, branch_factor=2, beam_width=2:
            # Level 0: 2 roots
            # Level 1: 2 beams * 2 children = 4
            # Level 2: 2 beams * 2 children = 4
            # Total = 2 + 4 + 4 = 10
            assert call_count == 10

    @pytest.mark.asyncio
    async def test_beam_search_returns_best_path(
        self, sample_problem, sample_config
    ):
        """Test that beam search returns a valid best path."""
        score_base = 6.0

        async def mock_gen_eval(problem, config=None, **kwargs):
            # Use modulo to stay in valid range
            return ScoredThought(
                thought=Thought(content="Node", reasoning="R"),
                evaluation=ThoughtEvaluation(
                    score=score_base, is_valid=True, feedback="OK"
                ),
            )

        with patch(
            "agentic_patterns.tree_of_thoughts.generate_and_evaluate",
            side_effect=mock_gen_eval,
        ):
            all_nodes, best_path = await beam_search(
                sample_problem, sample_config
            )

            assert len(best_path) > 0
            # Path should trace from root to leaf
            assert best_path[0].depth == 0


class TestSynthesizeSolution:
    """Test solution synthesis."""

    @pytest.fixture
    def sample_problem(self):
        return ProblemStatement(description="Game of 24")

    @pytest.fixture
    def sample_path(self):
        return [
            ThoughtNode(
                id="0.0",
                depth=0,
                scored_thought=ScoredThought(
                    thought=Thought(content="Step 1", reasoning="R1"),
                    evaluation=ThoughtEvaluation(
                        score=7.0, is_valid=True, feedback="OK"
                    ),
                ),
            ),
            ThoughtNode(
                id="0.0.0",
                depth=1,
                scored_thought=ScoredThought(
                    thought=Thought(content="Step 2", reasoning="R2"),
                    evaluation=ThoughtEvaluation(
                        score=8.0, is_valid=True, feedback="Good"
                    ),
                ),
                parent_id="0.0",
            ),
        ]

    @pytest.mark.asyncio
    async def test_synthesize_solution_returns_solution(
        self, sample_problem, sample_path
    ):
        """Test that synthesize_solution returns a SynthesizedSolution."""
        mock_solution = SynthesizedSolution(
            solution="8 * 3 = 24",
            confidence=0.9,
            reasoning="Combined steps to reach 24",
        )

        async def mock_run(prompt, deps):
            result = MagicMock()
            result.output = mock_solution
            return result

        with patch(
            "agentic_patterns.tree_of_thoughts.synthesis_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(side_effect=mock_run)

            solution = await synthesize_solution(sample_problem, sample_path)

            assert isinstance(solution, SynthesizedSolution)
            assert solution.solution == "8 * 3 = 24"

    @pytest.mark.asyncio
    async def test_synthesize_solution_passes_context(
        self, sample_problem, sample_path
    ):
        """Test that problem and path are passed through deps."""
        captured_deps = None

        async def mock_run(prompt, deps):
            nonlocal captured_deps
            captured_deps = deps
            result = MagicMock()
            result.output = SynthesizedSolution(
                solution="X", confidence=0.5, reasoning="Y"
            )
            return result

        with patch(
            "agentic_patterns.tree_of_thoughts.synthesis_agent"
        ) as mock_agent:
            mock_agent.run = AsyncMock(side_effect=mock_run)

            await synthesize_solution(sample_problem, sample_path)

            assert captured_deps is not None
            assert isinstance(captured_deps, SynthesisContext)
            assert captured_deps.problem == sample_problem
            assert len(captured_deps.path) == 2


class TestRunTreeOfThoughts:
    """Test full Tree of Thoughts execution."""

    @pytest.fixture
    def sample_problem(self):
        return ProblemStatement(
            description="Game of 24: Use 4, 7, 8, 8",
            constraints=["Result must equal 24"],
        )

    @pytest.mark.asyncio
    async def test_run_tree_of_thoughts_default_config(self, sample_problem):
        """Test with default configuration."""
        mock_scored = ScoredThought(
            thought=Thought(content="Approach", reasoning="R"),
            evaluation=ThoughtEvaluation(
                score=7.0, is_valid=True, feedback="OK"
            ),
        )
        mock_solution = SynthesizedSolution(
            solution="Answer", confidence=0.8, reasoning="Derived"
        )

        async def mock_gen_eval(problem, config=None, **kwargs):
            return mock_scored

        async def mock_synth_run(prompt, deps):
            result = MagicMock()
            result.output = mock_solution
            return result

        with (
            patch(
                "agentic_patterns.tree_of_thoughts.generate_and_evaluate",
                side_effect=mock_gen_eval,
            ),
            patch(
                "agentic_patterns.tree_of_thoughts.synthesis_agent"
            ) as mock_synth,
        ):
            mock_synth.run = AsyncMock(side_effect=mock_synth_run)

            result = await run_tree_of_thoughts(sample_problem)

            assert isinstance(result, TreeExplorationResult)
            assert result.problem == sample_problem
            assert result.config.max_depth == 3  # Default

    @pytest.mark.asyncio
    async def test_run_tree_of_thoughts_custom_config(self, sample_problem):
        """Test with custom configuration."""
        config = TreeConfig(max_depth=2, branch_factor=2)

        mock_scored = ScoredThought(
            thought=Thought(content="X", reasoning="R"),
            evaluation=ThoughtEvaluation(
                score=7.0, is_valid=True, feedback="OK"
            ),
        )
        mock_solution = SynthesizedSolution(
            solution="Y", confidence=0.7, reasoning="Z"
        )

        async def mock_gen_eval(problem, config=None, **kwargs):
            return mock_scored

        async def mock_synth_run(prompt, deps):
            result = MagicMock()
            result.output = mock_solution
            return result

        with (
            patch(
                "agentic_patterns.tree_of_thoughts.generate_and_evaluate",
                side_effect=mock_gen_eval,
            ),
            patch(
                "agentic_patterns.tree_of_thoughts.synthesis_agent"
            ) as mock_synth,
        ):
            mock_synth.run = AsyncMock(side_effect=mock_synth_run)

            result = await run_tree_of_thoughts(sample_problem, config)

            assert result.config == config
            assert result.config.max_depth == 2

    @pytest.mark.asyncio
    async def test_run_tree_of_thoughts_counts_nodes(self, sample_problem):
        """Test that node counts are accurate."""
        # max_depth=0 means only roots, no expansion
        config = TreeConfig(
            max_depth=1, branch_factor=3, prune_threshold=5.0, beam_width=2
        )

        # Scores for roots: 7.0, 4.0, 6.0 (one pruned at 4.0)
        # Then depth 1: 2 unpruned roots * 3 children = 6 nodes (all at 7.0)
        # Total = 3 + 6 = 9 nodes, 1 pruned
        scores = [7.0, 4.0, 6.0] + [7.0] * 6
        score_iter = iter(scores)

        async def mock_gen_eval(problem, config=None, **kwargs):
            score = next(score_iter, 7.0)
            return ScoredThought(
                thought=Thought(content="X", reasoning="R"),
                evaluation=ThoughtEvaluation(
                    score=score, is_valid=True, feedback="OK"
                ),
            )

        mock_solution = SynthesizedSolution(
            solution="Y", confidence=0.7, reasoning="Z"
        )

        async def mock_synth_run(prompt, deps):
            result = MagicMock()
            result.output = mock_solution
            return result

        with (
            patch(
                "agentic_patterns.tree_of_thoughts.generate_and_evaluate",
                side_effect=mock_gen_eval,
            ),
            patch(
                "agentic_patterns.tree_of_thoughts.synthesis_agent"
            ) as mock_synth,
        ):
            mock_synth.run = AsyncMock(side_effect=mock_synth_run)

            result = await run_tree_of_thoughts(sample_problem, config)

            # 3 roots + 6 children (2 beams * 3 branches) = 9
            assert result.nodes_explored == 9
            assert result.nodes_pruned == 1  # One root below threshold

    @pytest.mark.asyncio
    async def test_run_tree_of_thoughts_has_best_path(self, sample_problem):
        """Test that result includes a best path."""
        config = TreeConfig(max_depth=1, branch_factor=2)

        mock_scored = ScoredThought(
            thought=Thought(content="X", reasoning="R"),
            evaluation=ThoughtEvaluation(
                score=7.0, is_valid=True, feedback="OK"
            ),
        )
        mock_solution = SynthesizedSolution(
            solution="Y", confidence=0.7, reasoning="Z"
        )

        async def mock_gen_eval(problem, config=None, **kwargs):
            return mock_scored

        async def mock_synth_run(prompt, deps):
            result = MagicMock()
            result.output = mock_solution
            return result

        with (
            patch(
                "agentic_patterns.tree_of_thoughts.generate_and_evaluate",
                side_effect=mock_gen_eval,
            ),
            patch(
                "agentic_patterns.tree_of_thoughts.synthesis_agent"
            ) as mock_synth,
        ):
            mock_synth.run = AsyncMock(side_effect=mock_synth_run)

            result = await run_tree_of_thoughts(sample_problem, config)

            assert len(result.best_path) > 0
            assert result.solution is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_tree_config_minimum_values(self):
        config = TreeConfig(
            max_depth=1,
            branch_factor=1,
            prune_threshold=0.0,
            beam_width=1,
        )
        assert config.max_depth == 1
        assert config.branch_factor == 1

    def test_thought_node_root_has_no_parent(self):
        node = ThoughtNode(
            id="0.0",
            depth=0,
            scored_thought=ScoredThought(
                thought=Thought(content="Root", reasoning="R"),
                evaluation=ThoughtEvaluation(
                    score=7.0, is_valid=True, feedback="OK"
                ),
            ),
        )
        assert node.parent_id is None

    def test_tree_exploration_result_empty_path(self):
        """Edge case: result with empty path."""
        problem = ProblemStatement(description="Test")
        config = TreeConfig()
        node = ThoughtNode(
            id="0.0",
            depth=0,
            scored_thought=ScoredThought(
                thought=Thought(content="X", reasoning="R"),
                evaluation=ThoughtEvaluation(
                    score=5.0, is_valid=True, feedback="OK"
                ),
            ),
        )
        solution = SynthesizedSolution(
            solution="Fallback", confidence=0.1, reasoning="No path"
        )

        # Should handle empty best_path gracefully
        result = TreeExplorationResult(
            problem=problem,
            config=config,
            all_nodes=[node],
            best_path=[],  # Empty path
            solution=solution,
            nodes_explored=1,
            nodes_pruned=0,
        )
        assert len(result.best_path) == 0

    @pytest.mark.asyncio
    async def test_expand_node_at_max_depth_returns_empty(self):
        """Nodes at max depth should not expand."""
        problem = ProblemStatement(description="Test")
        config = TreeConfig(max_depth=2)
        node = ThoughtNode(
            id="0.0.0",
            depth=2,  # At max
            scored_thought=ScoredThought(
                thought=Thought(content="Deep", reasoning="R"),
                evaluation=ThoughtEvaluation(
                    score=9.0, is_valid=True, feedback="OK"
                ),
            ),
        )

        children = await expand_node(problem, node, config)
        assert children == []
