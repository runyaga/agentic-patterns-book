"""Tests for the Parallelization Pattern implementation."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns.parallelization import DocumentSummary
from agentic_patterns.parallelization import ReducedSummary
from agentic_patterns.parallelization import SectionResult
from agentic_patterns.parallelization import SynthesizedResult
from agentic_patterns.parallelization import VoteResult
from agentic_patterns.parallelization import VotingOutcome
from agentic_patterns.parallelization import run_map_reduce
from agentic_patterns.parallelization import run_sectioning
from agentic_patterns.parallelization import run_voting


class TestModels:
    """Test Pydantic model validation."""

    def test_section_result_valid(self):
        result = SectionResult(
            section_name="History",
            content="AI began in the 1950s...",
            key_points=["Turing Test", "Expert Systems"],
        )
        assert result.section_name == "History"
        assert len(result.key_points) == 2

    def test_section_result_empty_key_points(self):
        result = SectionResult(
            section_name="Test",
            content="Content here",
        )
        assert result.key_points == []

    def test_synthesized_result_valid(self):
        result = SynthesizedResult(
            summary="AI has evolved significantly...",
            all_key_points=["Point 1", "Point 2", "Point 3"],
            section_count=3,
        )
        assert result.section_count == 3
        assert len(result.all_key_points) == 3

    def test_vote_result_valid(self):
        result = VoteResult(
            answer="Yes",
            confidence=0.85,
            reasoning="Python has great ML libraries",
        )
        assert result.answer == "Yes"
        assert result.confidence == 0.85

    def test_vote_result_confidence_bounds(self):
        # Valid at boundaries
        VoteResult(answer="Yes", confidence=0.0, reasoning="Uncertain")
        VoteResult(answer="Yes", confidence=1.0, reasoning="Certain")

        # Invalid
        with pytest.raises(ValueError):
            VoteResult(answer="Yes", confidence=1.5, reasoning="Invalid")

        with pytest.raises(ValueError):
            VoteResult(answer="Yes", confidence=-0.1, reasoning="Invalid")

    def test_voting_outcome_valid(self):
        outcome = VotingOutcome(
            winning_answer="Yes",
            vote_count=2,
            total_votes=3,
            all_answers=["Yes", "Yes", "No"],
        )
        assert outcome.winning_answer == "Yes"
        assert outcome.vote_count == 2

    def test_document_summary_valid(self):
        summary = DocumentSummary(
            doc_id="doc1",
            summary="This document discusses...",
            word_count=500,
        )
        assert summary.doc_id == "doc1"
        assert summary.word_count == 500

    def test_reduced_summary_valid(self):
        result = ReducedSummary(
            combined_summary="Overview of all documents...",
            total_documents=3,
            total_words=1500,
        )
        assert result.total_documents == 3
        assert result.total_words == 1500


class TestSectioning:
    """Test the sectioning parallelization pattern."""

    @pytest.fixture
    def mock_section_results(self):
        return [
            SectionResult(
                section_name="History",
                content="AI started in 1950s",
                key_points=["Turing", "Dartmouth"],
            ),
            SectionResult(
                section_name="Current",
                content="ML is widely used",
                key_points=["Deep Learning", "NLP"],
            ),
            SectionResult(
                section_name="Future",
                content="AGI is the goal",
                key_points=["AGI", "Ethics"],
            ),
        ]

    @pytest.fixture
    def mock_synthesis_result(self):
        return SynthesizedResult(
            summary="AI has evolved from theory to practice",
            all_key_points=["Turing", "Deep Learning", "AGI"],
            section_count=3,
        )

    @pytest.mark.asyncio
    async def test_run_sectioning_parallel_execution(
        self, mock_section_results, mock_synthesis_result
    ):
        """Test that sections run in parallel and are synthesized."""
        call_order = []

        async def mock_section_run(prompt):
            section_name = prompt.split("Section focus: ")[1].split("\n")[0]
            call_order.append(f"section_{section_name}")
            # Find matching result
            for r in mock_section_results:
                if r.section_name == section_name:
                    result = MagicMock()
                    result.output = r
                    return result
            result = MagicMock()
            result.output = mock_section_results[0]
            return result

        async def mock_synthesis_run(prompt):
            call_order.append("synthesis")
            result = MagicMock()
            result.output = mock_synthesis_result
            return result

        with (
            patch(
                "agentic_patterns.parallelization.section_agent"
            ) as mock_section,
            patch(
                "agentic_patterns.parallelization.synthesis_agent"
            ) as mock_synthesis,
        ):
            mock_section.run = AsyncMock(side_effect=mock_section_run)
            mock_synthesis.run = AsyncMock(side_effect=mock_synthesis_run)

            result = await run_sectioning(
                topic="AI",
                sections=["History", "Current", "Future"],
            )

            assert isinstance(result, SynthesizedResult)
            assert result.section_count == 3
            # Synthesis should be called after all sections
            assert call_order[-1] == "synthesis"
            assert mock_section.run.call_count == 3

    @pytest.mark.asyncio
    async def test_run_sectioning_combines_key_points(
        self, mock_section_results, mock_synthesis_result
    ):
        """Test that key points from all sections are passed to synthesis."""
        captured_prompt = None

        async def mock_synthesis_run(prompt):
            nonlocal captured_prompt
            captured_prompt = prompt
            result = MagicMock()
            result.output = mock_synthesis_result
            return result

        with (
            patch(
                "agentic_patterns.parallelization.section_agent"
            ) as mock_section,
            patch(
                "agentic_patterns.parallelization.synthesis_agent"
            ) as mock_synthesis,
        ):
            # Return results in order
            results = iter(mock_section_results)

            async def section_run(prompt):
                result = MagicMock()
                result.output = next(results)
                return result

            mock_section.run = AsyncMock(side_effect=section_run)
            mock_synthesis.run = AsyncMock(side_effect=mock_synthesis_run)

            await run_sectioning(
                topic="AI",
                sections=["History", "Current", "Future"],
            )

            # Verify synthesis received all section data
            assert "History" in captured_prompt
            assert "Current" in captured_prompt
            assert "Future" in captured_prompt


class TestVoting:
    """Test the voting parallelization pattern."""

    @pytest.fixture
    def mock_vote_results_unanimous(self):
        return [
            VoteResult(answer="Yes", confidence=0.9, reasoning="Reason 1"),
            VoteResult(answer="Yes", confidence=0.85, reasoning="Reason 2"),
            VoteResult(answer="Yes", confidence=0.95, reasoning="Reason 3"),
        ]

    @pytest.fixture
    def mock_vote_results_split(self):
        return [
            VoteResult(answer="Yes", confidence=0.9, reasoning="Reason 1"),
            VoteResult(answer="No", confidence=0.7, reasoning="Reason 2"),
            VoteResult(answer="Yes", confidence=0.8, reasoning="Reason 3"),
        ]

    @pytest.mark.asyncio
    async def test_run_voting_unanimous(self, mock_vote_results_unanimous):
        """Test voting with unanimous agreement."""
        results_iter = iter(mock_vote_results_unanimous)

        async def mock_vote_run(prompt):
            result = MagicMock()
            result.output = next(results_iter)
            return result

        with patch(
            "agentic_patterns.parallelization.voting_agent"
        ) as mock_voter:
            mock_voter.run = AsyncMock(side_effect=mock_vote_run)

            outcome = await run_voting("Is Python good?", num_voters=3)

            assert outcome.winning_answer == "Yes"
            assert outcome.vote_count == 3
            assert outcome.total_votes == 3

    @pytest.mark.asyncio
    async def test_run_voting_majority(self, mock_vote_results_split):
        """Test voting with majority (not unanimous)."""
        results_iter = iter(mock_vote_results_split)

        async def mock_vote_run(prompt):
            result = MagicMock()
            result.output = next(results_iter)
            return result

        with patch(
            "agentic_patterns.parallelization.voting_agent"
        ) as mock_voter:
            mock_voter.run = AsyncMock(side_effect=mock_vote_run)

            outcome = await run_voting("Is Python good?", num_voters=3)

            assert outcome.winning_answer == "Yes"
            assert outcome.vote_count == 2
            assert outcome.total_votes == 3
            assert "No" in outcome.all_answers

    @pytest.mark.asyncio
    async def test_run_voting_correct_voter_count(self):
        """Test that correct number of voters are invoked."""
        mock_result = VoteResult(
            answer="Yes", confidence=0.9, reasoning="Test"
        )

        async def mock_vote_run(prompt):
            result = MagicMock()
            result.output = mock_result
            return result

        with patch(
            "agentic_patterns.parallelization.voting_agent"
        ) as mock_voter:
            mock_voter.run = AsyncMock(side_effect=mock_vote_run)

            await run_voting("Question?", num_voters=5)

            assert mock_voter.run.call_count == 5


class TestMapReduce:
    """Test the map-reduce parallelization pattern."""

    @pytest.fixture
    def mock_doc_summaries(self):
        return [
            DocumentSummary(
                doc_id="doc1", summary="Summary of doc1", word_count=100
            ),
            DocumentSummary(
                doc_id="doc2", summary="Summary of doc2", word_count=200
            ),
            DocumentSummary(
                doc_id="doc3", summary="Summary of doc3", word_count=150
            ),
        ]

    @pytest.fixture
    def mock_reduced_result(self):
        return ReducedSummary(
            combined_summary="Combined overview",
            total_documents=3,
            total_words=0,  # Will be updated
        )

    @pytest.mark.asyncio
    async def test_run_map_reduce_maps_all_documents(
        self, mock_doc_summaries, mock_reduced_result
    ):
        """Test that all documents are mapped in parallel."""
        summaries_iter = iter(mock_doc_summaries)

        async def mock_map_run(prompt):
            result = MagicMock()
            result.output = next(summaries_iter)
            return result

        async def mock_reduce_run(prompt):
            result = MagicMock()
            result.output = mock_reduced_result
            return result

        with (
            patch("agentic_patterns.parallelization.map_agent") as mock_map,
            patch(
                "agentic_patterns.parallelization.reduce_agent"
            ) as mock_reduce,
        ):
            mock_map.run = AsyncMock(side_effect=mock_map_run)
            mock_reduce.run = AsyncMock(side_effect=mock_reduce_run)

            documents = [
                ("doc1", "Content 1"),
                ("doc2", "Content 2"),
                ("doc3", "Content 3"),
            ]

            result = await run_map_reduce(documents)

            assert mock_map.run.call_count == 3
            assert mock_reduce.run.call_count == 1
            assert isinstance(result, ReducedSummary)

    @pytest.mark.asyncio
    async def test_run_map_reduce_calculates_total_words(
        self, mock_doc_summaries, mock_reduced_result
    ):
        """Test that total word count is calculated from mapped results."""
        summaries_iter = iter(mock_doc_summaries)

        async def mock_map_run(prompt):
            result = MagicMock()
            result.output = next(summaries_iter)
            return result

        async def mock_reduce_run(prompt):
            result = MagicMock()
            result.output = mock_reduced_result
            return result

        with (
            patch("agentic_patterns.parallelization.map_agent") as mock_map,
            patch(
                "agentic_patterns.parallelization.reduce_agent"
            ) as mock_reduce,
        ):
            mock_map.run = AsyncMock(side_effect=mock_map_run)
            mock_reduce.run = AsyncMock(side_effect=mock_reduce_run)

            documents = [
                ("doc1", "Content 1"),
                ("doc2", "Content 2"),
                ("doc3", "Content 3"),
            ]

            result = await run_map_reduce(documents)

            # 100 + 200 + 150 = 450
            assert result.total_words == 450

    @pytest.mark.asyncio
    async def test_run_map_reduce_passes_summaries_to_reduce(
        self, mock_doc_summaries, mock_reduced_result
    ):
        """Test that all summaries are passed to the reduce step."""
        summaries_iter = iter(mock_doc_summaries)
        captured_reduce_prompt = None

        async def mock_map_run(prompt):
            result = MagicMock()
            result.output = next(summaries_iter)
            return result

        async def mock_reduce_run(prompt):
            nonlocal captured_reduce_prompt
            captured_reduce_prompt = prompt
            result = MagicMock()
            result.output = mock_reduced_result
            return result

        with (
            patch("agentic_patterns.parallelization.map_agent") as mock_map,
            patch(
                "agentic_patterns.parallelization.reduce_agent"
            ) as mock_reduce,
        ):
            mock_map.run = AsyncMock(side_effect=mock_map_run)
            mock_reduce.run = AsyncMock(side_effect=mock_reduce_run)

            documents = [
                ("doc1", "Content 1"),
                ("doc2", "Content 2"),
                ("doc3", "Content 3"),
            ]

            await run_map_reduce(documents)

            # Verify all doc summaries were passed to reduce
            assert "doc1" in captured_reduce_prompt
            assert "doc2" in captured_reduce_prompt
            assert "doc3" in captured_reduce_prompt
            assert "Summary of doc1" in captured_reduce_prompt


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_voting_outcome_all_same_votes(self):
        outcome = VotingOutcome(
            winning_answer="Yes",
            vote_count=5,
            total_votes=5,
            all_answers=["Yes"] * 5,
        )
        assert outcome.vote_count == outcome.total_votes

    def test_synthesized_result_empty_key_points(self):
        result = SynthesizedResult(
            summary="No key points",
            all_key_points=[],
            section_count=1,
        )
        assert len(result.all_key_points) == 0

    def test_reduced_summary_single_document(self):
        result = ReducedSummary(
            combined_summary="Single doc",
            total_documents=1,
            total_words=100,
        )
        assert result.total_documents == 1
