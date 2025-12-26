"""
Parallelization Pattern Implementation.

Based on the Agentic Design Patterns book Chapter 3:
Execute multiple independent sub-tasks concurrently to reduce latency.

Three approaches demonstrated:
1. Sectioning - Divide work into independent sections run in parallel
2. Voting - Run same task multiple times and aggregate results
3. Map-Reduce - Map inputs to parallel workers, then reduce outputs
"""

import asyncio
from collections import Counter

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent

from agentic_patterns._models import get_model


class SectionResult(BaseModel):
    """Result from a single section of parallel work."""

    section_name: str = Field(description="Name of the section")
    content: str = Field(description="Content produced by this section")
    key_points: list[str] = Field(
        default_factory=list, description="Key points extracted"
    )


class SynthesizedResult(BaseModel):
    """Final synthesized result from multiple sections."""

    summary: str = Field(description="Combined summary of all sections")
    all_key_points: list[str] = Field(description="Merged key points")
    section_count: int = Field(description="Number of sections processed")


class VoteResult(BaseModel):
    """Result from a single voting agent."""

    answer: str = Field(description="The agent's answer")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the answer"
    )
    reasoning: str = Field(description="Reasoning behind the answer")


class VotingOutcome(BaseModel):
    """Aggregated outcome from voting."""

    winning_answer: str = Field(description="The most common answer")
    vote_count: int = Field(description="Number of votes for winner")
    total_votes: int = Field(description="Total number of votes")
    all_answers: list[str] = Field(description="All answers received")


class DocumentSummary(BaseModel):
    """Summary of a single document (map phase)."""

    doc_id: str = Field(description="Document identifier")
    summary: str = Field(description="Summary of the document")
    word_count: int = Field(description="Approximate word count")


class ReducedSummary(BaseModel):
    """Final reduced summary from all documents."""

    combined_summary: str = Field(description="Combined summary")
    total_documents: int = Field(description="Number of documents processed")
    total_words: int = Field(description="Total words across all documents")


# Initialize the model
model = get_model()

# Section worker agent
section_agent = Agent(
    model,
    system_prompt=(
        "You are a research assistant. Given a topic and section focus, "
        "produce relevant content and extract key points. "
        "Be concise but thorough."
    ),
    output_type=SectionResult,
)

# Synthesis agent for combining sections
synthesis_agent = Agent(
    model,
    system_prompt=(
        "You are a synthesis expert. Given multiple section results, "
        "combine them into a coherent summary. "
        "Merge key points and eliminate redundancy."
    ),
    output_type=SynthesizedResult,
)

# Voting agent
voting_agent = Agent(
    model,
    system_prompt=(
        "You are an expert analyst. Answer the question with confidence. "
        "Provide your reasoning. Be decisive."
    ),
    output_type=VoteResult,
)

# Map agent (document summarizer)
map_agent = Agent(
    model,
    system_prompt=(
        "You are a document summarizer. Summarize the given document "
        "concisely. Estimate the word count of the original."
    ),
    output_type=DocumentSummary,
)

# Reduce agent (combines summaries)
reduce_agent = Agent(
    model,
    system_prompt=(
        "You are a synthesis expert. Combine multiple document summaries "
        "into a single coherent overview. Preserve key information."
    ),
    output_type=ReducedSummary,
)


async def run_sectioning(
    topic: str,
    sections: list[str],
) -> SynthesizedResult:
    """
    Run sectioning parallelization pattern.

    Divides work into independent sections that run concurrently,
    then synthesizes the results.

    Args:
        topic: The main topic to research.
        sections: List of section focuses to explore in parallel.

    Returns:
        SynthesizedResult combining all section outputs.
    """
    print(f"Sectioning: Running {len(sections)} sections in parallel...")

    async def process_section(section_name: str) -> SectionResult:
        result = await section_agent.run(
            f"Topic: {topic}\nSection focus: {section_name}\n"
            f"Produce content for this section."
        )
        return result.output

    # Run all sections in parallel
    section_results = await asyncio.gather(
        *[process_section(s) for s in sections]
    )

    print(f"  All {len(section_results)} sections complete. Synthesizing...")

    # Synthesize results
    sections_text = "\n\n".join(
        f"Section: {r.section_name}\n"
        f"Content: {r.content}\n"
        f"Key Points: {', '.join(r.key_points)}"
        for r in section_results
    )

    synthesis_result = await synthesis_agent.run(
        f"Synthesize these section results:\n\n{sections_text}"
    )

    print("  Synthesis complete.")
    return synthesis_result.output


async def run_voting(
    question: str,
    num_voters: int = 3,
) -> VotingOutcome:
    """
    Run voting parallelization pattern.

    Runs the same question through multiple agents in parallel
    and aggregates their answers by majority vote.

    Args:
        question: The question to answer.
        num_voters: Number of parallel voters (default: 3).

    Returns:
        VotingOutcome with the winning answer and vote counts.
    """
    print(f"Voting: Running {num_voters} voters in parallel...")

    async def get_vote(voter_id: int) -> VoteResult:
        result = await voting_agent.run(
            f"Voter {voter_id}: Answer this question:\n{question}"
        )
        return result.output

    # Run all voters in parallel
    vote_results = await asyncio.gather(
        *[get_vote(i) for i in range(num_voters)]
    )

    print(f"  All {len(vote_results)} votes received. Counting...")

    # Count votes (simple majority)
    all_answers = [v.answer.lower().strip() for v in vote_results]
    vote_counts = Counter(all_answers)
    winner, count = vote_counts.most_common(1)[0]

    # Find original case version
    for v in vote_results:
        if v.answer.lower().strip() == winner:
            winner = v.answer
            break

    outcome = VotingOutcome(
        winning_answer=winner,
        vote_count=count,
        total_votes=len(vote_results),
        all_answers=[v.answer for v in vote_results],
    )

    print(f"  Winner: '{winner}' with {count}/{len(vote_results)} votes")
    return outcome


async def run_map_reduce(
    documents: list[tuple[str, str]],
) -> ReducedSummary:
    """
    Run map-reduce parallelization pattern.

    Maps each document to a summarizer in parallel (map phase),
    then combines all summaries (reduce phase).

    Args:
        documents: List of (doc_id, content) tuples.

    Returns:
        ReducedSummary combining all document summaries.
    """
    print(f"Map-Reduce: Mapping {len(documents)} documents in parallel...")

    async def map_document(doc_id: str, content: str) -> DocumentSummary:
        result = await map_agent.run(
            f"Document ID: {doc_id}\n\nContent:\n{content}"
        )
        return result.output

    # Map phase: summarize all documents in parallel
    summaries = await asyncio.gather(
        *[map_document(doc_id, content) for doc_id, content in documents]
    )

    print(f"  Map complete. Reducing {len(summaries)} summaries...")

    # Reduce phase: combine summaries
    summaries_text = "\n\n".join(
        f"Document {s.doc_id}:\n{s.summary}" for s in summaries
    )
    total_words = sum(s.word_count for s in summaries)

    reduce_result = await reduce_agent.run(
        f"Combine these document summaries into a single overview:\n\n"
        f"{summaries_text}\n\nTotal documents: {len(summaries)}"
    )

    result = reduce_result.output
    result.total_words = total_words

    print("  Reduce complete.")
    return result


if __name__ == "__main__":

    async def main() -> None:
        # Demo 1: Sectioning
        print("\n" + "=" * 60)
        print("DEMO 1: Sectioning Pattern")
        print("=" * 60)
        result1 = await run_sectioning(
            topic="Artificial Intelligence",
            sections=["History", "Current Applications", "Future Trends"],
        )
        print(f"\nSummary: {result1.summary[:200]}...")
        print(f"Key Points: {len(result1.all_key_points)}")

        # Demo 2: Voting
        print("\n" + "=" * 60)
        print("DEMO 2: Voting Pattern")
        print("=" * 60)
        result2 = await run_voting(
            question="Is Python a good language for AI development?",
            num_voters=3,
        )
        print(f"\nWinner: {result2.winning_answer}")
        print(f"Votes: {result2.vote_count}/{result2.total_votes}")

        # Demo 3: Map-Reduce
        print("\n" + "=" * 60)
        print("DEMO 3: Map-Reduce Pattern")
        print("=" * 60)
        docs = [
            ("doc1", "Python is a versatile programming language..."),
            ("doc2", "Machine learning requires large datasets..."),
            ("doc3", "Neural networks are inspired by the brain..."),
        ]
        result3 = await run_map_reduce(docs)
        print(f"\nCombined: {result3.combined_summary[:200]}...")
        print(f"Documents: {result3.total_documents}")

    asyncio.run(main())
