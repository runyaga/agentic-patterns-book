# Chapter 3: Parallelization

Execute independent sub-tasks concurrently (`asyncio.gather`) to reduce latency.

## Implementation

Source: `src/agentic_patterns/parallelization.py`

### 1. Sectioning (Task Division)

Divide work into independent sections.

```python
async def run_sectioning(topic: str, sections: list[str]) -> SynthesizedResult:
    async def process_section(section_name: str) -> SectionResult:
        result = await section_agent.run(
            f"Topic: {topic}\nSection focus: {section_name}"
        )
        return result.output

    # Run in parallel
    section_results = await asyncio.gather(*[process_section(s) for s in sections])

    # Synthesize
    return await synthesis_agent.run(format_sections(section_results))
```

### 2. Voting (Majority Consensus)

Run same task multiple times for reliability.

```python
async def run_voting(question: str, num_voters: int = 3) -> VotingOutcome:
    async def get_vote(voter_id: int) -> VoteResult:
        # Each voter is an independent agent call
        return (await voting_agent.run(f"Voter {voter_id}: {question}")).output

    vote_results = await asyncio.gather(*[get_vote(i) for i in range(num_voters)])

    # Aggregation logic (e.g., Counter(answers).most_common(1))
    ...
```

### 3. Map-Reduce (Batch Processing)

Process items in parallel (Map), then combine (Reduce).

```python
async def run_map_reduce(documents: list[tuple[str, str]]) -> ReducedSummary:
    # Map: Summarize each doc
    summaries = await asyncio.gather(
        *[map_agent.run(f"Doc: {d}\n{c}") for d, c in documents]
    )

    # Reduce: Combine summaries
    return await reduce_agent.run(format_summaries(summaries))
```

## Use Cases

- **Sectioning**: Research (History, Pros/Cons), Content generation (Intro, Body, Conclusion).
- **Voting**: Fact-checking, Content safety classification, Creative brainstorming (best of N).
- **Map-Reduce**: Log analysis, Document summarization, Batch data extraction.

## When to Use

- **Sectioning**: Task divides into distinct, independent sub-topics.
- **Voting**: High accuracy needed; models may hallucinate or vary.
- **Map-Reduce**: Large datasets where items can be processed individually first.

## Example

```bash
.venv/bin/python -m agentic_patterns.parallelization
```
