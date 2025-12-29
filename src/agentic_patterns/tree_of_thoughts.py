"""
Tree of Thoughts Pattern (Multi-level Exploration with Pruning).

Based on the Agentic Design Patterns book Chapter 17b:
Multi-level tree exploration extending Thought Candidates (17a).

This pattern extends Best-of-N sampling to a tree structure, allowing
recursive exploration of solution paths with pruning of low-scoring
branches and beam search to focus on promising directions.

Key concepts:
- Tree Structure: Thoughts form a tree with parent-child relationships
- Beam Search: Keep top-k nodes at each level for expansion
- Pruning: Discard branches below a score threshold
- Synthesis: Combine best path into final solution

Extends: thought_candidates.py (17a)
"""

import asyncio
from dataclasses import dataclass

import logfire
from pydantic import BaseModel
from pydantic import Field
from pydantic import computed_field
from pydantic_ai import Agent
from pydantic_ai import RunContext

from agentic_patterns._models import get_strong_model
from agentic_patterns.thought_candidates import OutputConfig
from agentic_patterns.thought_candidates import ProblemStatement
from agentic_patterns.thought_candidates import ScoredThought
from agentic_patterns.thought_candidates import generate_and_evaluate


# --8<-- [start:models]
class ThoughtNode(BaseModel):
    """A node in the thought tree - wraps ScoredThought with structure."""

    id: str = Field(description="Unique node identifier (e.g., '0.1.2')")
    depth: int = Field(ge=0, description="Depth in tree (0 = root)")
    scored_thought: ScoredThought
    parent_id: str | None = Field(default=None, description="Parent node ID")
    children_ids: list[str] = Field(
        default_factory=list, description="Child node IDs"
    )
    is_pruned: bool = Field(
        default=False, description="Whether this branch was pruned"
    )

    @computed_field
    @property
    def score(self) -> float:
        """Expose score from underlying ScoredThought."""
        return self.scored_thought.score


class TreeConfig(BaseModel):
    """Configuration for tree exploration - all validated."""

    max_depth: int = Field(
        default=3, ge=1, le=10, description="Maximum tree depth"
    )
    branch_factor: int = Field(
        default=3, ge=1, le=10, description="Candidates per node"
    )
    prune_threshold: float = Field(
        default=5.0, ge=0.0, le=10.0, description="Minimum score to expand"
    )
    beam_width: int = Field(
        default=3, ge=1, description="Top-k nodes to expand per level"
    )


class SynthesizedSolution(BaseModel):
    """Final solution synthesized from the best path."""

    solution: str = Field(description="The final synthesized answer")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the solution"
    )
    reasoning: str = Field(description="How the solution was derived")


class TreeExplorationResult(BaseModel):
    """Complete result from Tree of Thoughts exploration."""

    problem: ProblemStatement
    config: TreeConfig
    all_nodes: list[ThoughtNode] = Field(description="All explored nodes")
    best_path: list[ThoughtNode] = Field(description="Best path to solution")
    solution: SynthesizedSolution
    nodes_explored: int = Field(description="Total nodes generated")
    nodes_pruned: int = Field(description="Nodes discarded by pruning")


@dataclass
class SynthesisContext:
    """Context for solution synthesis."""

    problem: ProblemStatement
    path: list[ThoughtNode]
    config: OutputConfig | None = None

    @property
    def output_config(self) -> OutputConfig:
        return self.config or OutputConfig()


# --8<-- [end:models]


# --8<-- [start:agents]
# Use strong model for synthesis (quality reasoning)
strong_model = get_strong_model()

synthesis_agent: Agent[SynthesisContext, SynthesizedSolution] = Agent(
    strong_model,  # Strong model for final synthesis
    system_prompt=(
        "You are a solution synthesizer. Given a sequence of reasoning "
        "steps, combine them into a final answer."
    ),
    deps_type=SynthesisContext,
    output_type=SynthesizedSolution,
)


@synthesis_agent.system_prompt
def inject_synthesis_context(ctx: RunContext[SynthesisContext]) -> str:
    """Inject problem, reasoning path, and output constraints."""
    cfg = ctx.deps.output_config
    steps = "\n".join(
        f"Step {i + 1}: {node.scored_thought.thought.content}"
        for i, node in enumerate(ctx.deps.path)
    )

    # Output constraints
    max_w = cfg.max_words
    output_rules = [f"Keep solution and reasoning each under {max_w} words."]
    if cfg.ascii_only:
        output_rules.append("Use plain ASCII only.")

    return (
        f"Problem: {ctx.deps.problem.description}\n\n"
        f"Reasoning path:\n{steps}\n\n"
        f"Output rules: {' '.join(output_rules)}\n\n"
        "Synthesize the final answer from this reasoning chain."
    )


# --8<-- [end:agents]


# --8<-- [start:patterns]
async def expand_node(
    problem: ProblemStatement,
    node: ThoughtNode,
    tree_config: TreeConfig,
    output_config: OutputConfig | None = None,
) -> list[ThoughtNode]:
    """
    Expand a node by generating child thoughts.

    Uses generate_and_evaluate from 17a to generate candidates,
    then wraps them as tree nodes with pruning.

    Args:
        problem: The structured problem statement.
        node: The parent node to expand.
        tree_config: Tree exploration configuration.
        output_config: Output constraints (word limits, ASCII-only, etc.).

    Returns:
        List of child ThoughtNodes (may include pruned nodes).
    """
    if node.is_pruned or node.depth >= tree_config.max_depth:
        return []

    with logfire.span("expand_node", node_id=node.id, depth=node.depth):
        # Build context from parent (typed, using model_copy)
        child_problem = problem.model_copy(
            update={"context": node.scored_thought.thought.content}
        )

        # Generate children in parallel using 17a's function
        tasks = [
            generate_and_evaluate(child_problem, output_config)
            for _ in range(tree_config.branch_factor)
        ]
        scored_thoughts = await asyncio.gather(*tasks)

        # Wrap as tree nodes with pruning
        children: list[ThoughtNode] = []
        for i, scored in enumerate(scored_thoughts):
            child = ThoughtNode(
                id=f"{node.id}.{i}",
                depth=node.depth + 1,
                scored_thought=scored,
                parent_id=node.id,
                is_pruned=scored.score < tree_config.prune_threshold,
            )
            children.append(child)
            node.children_ids.append(child.id)
            logfire.info(
                "Child generated",
                child_id=child.id,
                score=child.score,
                pruned=child.is_pruned,
            )

        return children


def trace_best_path(nodes: list[ThoughtNode]) -> list[ThoughtNode]:
    """
    Trace back from the highest-scoring leaf to root.

    Args:
        nodes: All nodes in the tree.

    Returns:
        List of nodes forming the best path (root to leaf).
    """
    if not nodes:
        return []

    node_map = {n.id: n for n in nodes}

    # Find best leaf (highest score among unpruned leaves)
    leaves = [n for n in nodes if not n.children_ids and not n.is_pruned]
    if not leaves:
        # Fallback to all nodes if no unpruned leaves
        leaves = [n for n in nodes if not n.children_ids]
    if not leaves:
        leaves = nodes

    best_leaf = max(leaves, key=lambda n: n.score)

    # Trace back to root
    path: list[ThoughtNode] = []
    current: ThoughtNode | None = best_leaf
    while current is not None:
        path.append(current)
        if current.parent_id:
            current = node_map.get(current.parent_id)
        else:
            current = None

    return list(reversed(path))


async def beam_search(
    problem: ProblemStatement,
    tree_config: TreeConfig,
    output_config: OutputConfig | None = None,
) -> tuple[list[ThoughtNode], list[ThoughtNode]]:
    """
    Perform beam search over the thought tree.

    Iteratively expands the top-k (beam_width) nodes at each level,
    pruning low-scoring branches.

    Args:
        problem: The structured problem statement.
        tree_config: Tree exploration configuration.
        output_config: Output constraints (word limits, ASCII-only, etc.).

    Returns:
        Tuple of (all_nodes, best_path_nodes).
    """
    with logfire.span("beam_search", config=tree_config.model_dump()):
        all_nodes: list[ThoughtNode] = []

        # Level 0: Generate root candidates using 17a
        with logfire.span("generate_roots", count=tree_config.branch_factor):
            logfire.info(
                f"Generating {tree_config.branch_factor} root candidates"
            )
            root_tasks = [
                generate_and_evaluate(problem, output_config)
                for _ in range(tree_config.branch_factor)
            ]
            root_scored = await asyncio.gather(*root_tasks)

        current_level: list[ThoughtNode] = []
        for i, scored in enumerate(root_scored):
            node = ThoughtNode(
                id=f"0.{i}",
                depth=0,
                scored_thought=scored,
                is_pruned=scored.score < tree_config.prune_threshold,
            )
            current_level.append(node)
            all_nodes.append(node)
            logfire.info(
                "Root node created",
                node_id=node.id,
                score=node.score,
                pruned=node.is_pruned,
            )

        # Explore levels until max depth
        for depth in range(1, tree_config.max_depth + 1):
            with logfire.span("explore_depth", depth=depth):
                # Select top-k (beam width) unpruned nodes to expand
                active = [n for n in current_level if not n.is_pruned]
                active.sort(key=lambda n: n.score, reverse=True)
                beam = active[: tree_config.beam_width]

                if not beam:
                    logfire.info("All nodes pruned, stopping", depth=depth)
                    break

                logfire.info(
                    "Expanding beam",
                    depth=depth,
                    beam_size=len(beam),
                    beam_scores=[n.score for n in beam],
                )

                # Expand each beam node in parallel
                expansion_tasks = [
                    expand_node(problem, n, tree_config, output_config)
                    for n in beam
                ]
                children_lists = await asyncio.gather(*expansion_tasks)

                current_level = []
                for children in children_lists:
                    current_level.extend(children)
                    all_nodes.extend(children)

        # Find best path
        best_path = trace_best_path(all_nodes)
        logfire.info(
            "Beam search complete",
            total_nodes=len(all_nodes),
            best_path_length=len(best_path),
        )

        return all_nodes, best_path


async def synthesize_solution(
    problem: ProblemStatement,
    path: list[ThoughtNode],
    output_config: OutputConfig | None = None,
) -> SynthesizedSolution:
    """
    Synthesize final solution from the best path.

    Args:
        problem: The structured problem statement.
        path: The best path through the tree.
        output_config: Output constraints (word limits, ASCII-only, etc.).

    Returns:
        SynthesizedSolution combining the reasoning steps.
    """
    with logfire.span("synthesize_solution", path_length=len(path)):
        ctx = SynthesisContext(
            problem=problem, path=path, config=output_config
        )
        result = await synthesis_agent.run("Synthesize the solution", deps=ctx)
        logfire.info(
            "Solution synthesized",
            confidence=result.output.confidence,
        )
        return result.output


async def run_tree_of_thoughts(
    problem: ProblemStatement,
    tree_config: TreeConfig | None = None,
    output_config: OutputConfig | None = None,
) -> TreeExplorationResult:
    """
    Run full Tree of Thoughts exploration.

    Extends 17a (Best-of-N) by recursively expanding the best candidates
    to form a tree, pruning low-scoring branches, and synthesizing
    the best path into a final solution.

    Args:
        problem: The structured problem statement.
        tree_config: Tree exploration configuration (uses defaults if None).
        output_config: Output constraints (word limits, ASCII-only, etc.).

    Returns:
        TreeExplorationResult with the full tree and solution.
    """
    tree_config = tree_config or TreeConfig()

    with logfire.span(
        "run_tree_of_thoughts",
        max_depth=tree_config.max_depth,
        branch_factor=tree_config.branch_factor,
        prune_threshold=tree_config.prune_threshold,
        beam_width=tree_config.beam_width,
    ):
        logfire.info("Starting Tree of Thoughts exploration")

        # Run beam search
        all_nodes, best_path = await beam_search(
            problem, tree_config, output_config
        )

        # Count pruned nodes
        nodes_pruned = sum(1 for n in all_nodes if n.is_pruned)

        logfire.info(
            "Tree exploration complete",
            nodes_explored=len(all_nodes),
            nodes_pruned=nodes_pruned,
            best_path_length=len(best_path),
        )

        # Synthesize solution from best path
        solution = await synthesize_solution(problem, best_path, output_config)

        return TreeExplorationResult(
            problem=problem,
            config=tree_config,
            all_nodes=all_nodes,
            best_path=best_path,
            solution=solution,
            nodes_explored=len(all_nodes),
            nodes_pruned=nodes_pruned,
        )


# --8<-- [end:patterns]


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree

    def render_tree(result: TreeExplorationResult, console: Console) -> None:
        """Render Tree of Thoughts results with rich visualization."""
        console.print()
        console.rule("[bold blue]Tree of Thoughts Exploration")
        console.print()

        # Problem panel
        console.print(
            Panel(
                f"[bold]{result.problem.description}[/bold]\n\n"
                f"[dim]Constraints: {', '.join(result.problem.constraints)}[/dim]",
                title="Problem",
                border_style="blue",
            )
        )

        # Config summary
        cfg = result.config
        console.print(
            f"\n[dim]Config: depth={cfg.max_depth}, "
            f"branch={cfg.branch_factor}, prune≥{cfg.prune_threshold}, "
            f"beam={cfg.beam_width}[/dim]\n"
        )

        # Build tree visualization
        best_path_ids = {n.id for n in result.best_path}
        node_map = {n.id: n for n in result.all_nodes}

        # Find root nodes (depth 0)
        roots = [n for n in result.all_nodes if n.depth == 0]

        tree = Tree(
            "[bold cyan]🌳 Thought Tree[/]",
            guide_style="dim",
        )

        def add_node_to_tree(
            parent_tree: Tree,
            node: ThoughtNode,
        ) -> None:
            """Recursively add nodes to the rich tree."""
            # Determine node style
            in_best = node.id in best_path_ids
            score = node.score
            if node.is_pruned:
                icon = "✗"
                style = "dim red strike"
            elif in_best:
                icon = "★"
                style = "bold green"
            elif score >= 7:
                icon = "●"
                style = "green"
            elif score >= 5:
                icon = "●"
                style = "yellow"
            else:
                icon = "●"
                style = "red"

            # Truncate content for display
            content = node.scored_thought.thought.content
            if len(content) > 60:
                content = content[:57] + "..."

            label = f"[{style}]{icon} [{score:.1f}] {content}[/]"
            branch = parent_tree.add(label)

            # Add children
            for child_id in node.children_ids:
                if child_id in node_map:
                    add_node_to_tree(branch, node_map[child_id])

        for root in sorted(roots, key=lambda n: n.score, reverse=True):
            add_node_to_tree(tree, root)

        console.print(tree)

        # Best path table
        console.print()
        path_table = Table(
            title="[bold green]★ Best Path[/]",
            show_header=True,
            header_style="bold cyan",
        )
        path_table.add_column("Step", justify="center", width=5)
        path_table.add_column("Score", justify="center", width=8)
        path_table.add_column("Thought", width=70)

        for i, node in enumerate(result.best_path):
            path_table.add_row(
                str(i + 1),
                f"[green]{node.score:.1f}[/]",
                node.scored_thought.thought.content,
            )

        console.print(path_table)

        # Solution panel
        console.print()
        console.print(
            Panel(
                f"[bold]{result.solution.solution}[/]\n\n"
                f"[bold]Confidence:[/] {result.solution.confidence:.0%}\n\n"
                f"[bold]Reasoning:[/]\n{result.solution.reasoning}",
                title="[bold green]Final Solution[/]",
                border_style="green",
            )
        )

        # Statistics
        console.print()
        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("Metric", style="dim")
        stats_table.add_column("Value", style="bold")
        stats_table.add_row("Nodes explored", str(result.nodes_explored))
        stats_table.add_row("Nodes pruned", str(result.nodes_pruned))
        stats_table.add_row("Best path length", str(len(result.best_path)))
        console.print(
            Panel(stats_table, title="Statistics", border_style="dim")
        )

    async def main() -> None:
        console = Console()
        console.print()
        console.rule("[bold]DEMO: Tree of Thoughts - Diagnostic Reasoning")

        # Diagnostic problem: explore hypotheses about a system issue
        problem = ProblemStatement(
            description=(
                "A web application is responding slowly. Response times "
                "increased from 200ms to 3 seconds over the past week. "
                "What are the most likely causes and how would you verify?"
            ),
            constraints=[
                "Consider database, application, and infrastructure layers",
                "Suggest specific diagnostic commands or metrics to check",
                "Prioritize most likely causes first",
            ],
        )

        tree_config = TreeConfig(
            max_depth=2,  # Reduced for faster demo
            branch_factor=2,
            prune_threshold=4.0,
            beam_width=2,
        )

        # Output constraints - tight for local models, expand for cloud
        output_config = OutputConfig(max_words=80, ascii_only=True)

        tc, oc = tree_config, output_config
        console.print(
            f"\n[dim]Tree: depth={tc.max_depth}, branch={tc.branch_factor}, "
            f"beam={tc.beam_width} | Output: max={oc.max_words}[/dim]\n"
        )
        result = await run_tree_of_thoughts(
            problem, tree_config, output_config
        )
        render_tree(result, console)

    asyncio.run(main())
