"""
Domain Exploration Pattern (The Cartographer).

Based on the Agentic Design Patterns book Chapter 21:
Autonomous discovery agent that crawls codebases and builds semantic maps.

Key capabilities:
- Structural Truth: AST-based extraction (zero-token) for code hierarchy
- Semantic Insight: LLM-based summarization of purpose and intent
- Stable Identity: Scoped IDs (e.g. `pkg.mod.Class`) to survive refactors
- Token Safety: "Dry Run" mode to preview crawl scope before LLM calls
- Persistence: Atomic JSON serialization with resumable frontier state

Flow diagram:

```mermaid
--8<-- [start:diagram]
stateDiagram-v2
    [*] --> ExploreNode: Initial path
    ExploreNode --> ExtractNode: Files to process
    ExtractNode --> MapNode: Entities found
    MapNode --> ExploreNode: More to explore
    MapNode --> CompleteNode: Boundary reached
    CompleteNode --> [*]: KnowledgeMap (JSON)
--8<-- [end:diagram]
```
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import re
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal

import logfire
import networkx as nx
from marktripy.parsers.markdown_it import MarkdownItParser
from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai import RunContext
from pydantic_ai.models import Model
from pydantic_graph import BaseNode
from pydantic_graph import End
from pydantic_graph import Graph
from pydantic_graph import GraphRunContext

from agentic_patterns._models import get_model

# --- Data Models ---


# --8<-- [start:models]
class SemanticEntity(BaseModel):
    """An entity discovered during exploration."""

    id: str = Field(description="Unique identifier (scoped hash)")
    name: str = Field(description="Entity name")
    entity_type: Literal[
        "module",
        "class",
        "function",
        "variable",
        "concept",
        "document",
        "api_endpoint",
        # Markdown entity types
        "section",
        "code_reference",
        "diagram",
    ] = Field(description="Type of entity")
    summary: str = Field(description="Brief description of the entity")
    location: str = Field(
        description="File path or URL where entity was found"
    )
    content_hash: str | None = Field(
        default=None, description="Hash of content for change detection"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticLink(BaseModel):
    """A relationship between two entities."""

    source_id: str
    target_id: str
    relationship: Literal[
        "imports",
        "calls",
        "defines",
        "references",
        "inherits",
        "implements",
        "contains",
        "depends_on",
    ]
    weight: float = 1.0


class TokenUsage(BaseModel):
    """Token usage from LLM calls."""

    input_tokens: int = Field(default=0, description="Input tokens consumed")
    output_tokens: int = Field(
        default=0, description="Output tokens generated"
    )

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Add two TokenUsage instances."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class ExplorationFrontier(BaseModel):
    """Tracks what has been explored and what remains."""

    explored: set[str] = Field(default_factory=set)
    pending: list[str] = Field(default_factory=list)
    depth_map: dict[str, int] = Field(default_factory=dict)


class KnowledgeMap(BaseModel):
    """The complete knowledge graph from exploration."""

    entities: list[SemanticEntity] = Field(default_factory=list)
    links: list[SemanticLink] = Field(default_factory=list)
    root_path: str
    last_updated: datetime = Field(default_factory=datetime.now)
    frontier_state: ExplorationFrontier | None = None
    # Token accounting
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    files_processed: int = Field(
        default=0, description="Number of files processed"
    )
    llm_calls: int = Field(
        default=0, description="Number of LLM extraction calls"
    )


class ExplorationBoundary(BaseModel):
    """Configuration for crawl boundaries."""

    max_depth: int = Field(default=5)
    max_files: int = Field(default=20)
    include_patterns: list[str] = Field(
        default_factory=lambda: ["**/*.py", "**/*.md"]
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "**/node_modules/**",
            "**/.git/**",
            "**/__pycache__/**",
        ]
    )
    dry_run: bool = Field(
        default=False, description="If True, skips LLM extraction (AST only)"
    )


def generate_entity_id(entity_type: str, scope: str, name: str) -> str:
    """
    Generate deterministic ID based on scope rather than file path.
    """
    content = f"{entity_type}:{scope}:{name}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _is_match(path: Path, pattern: str) -> bool:
    """
    Check if a path matches a glob pattern.

    Handles both standard glob matching and recursive patterns like '**/*.py'.
    """
    path_str = str(path)
    if path.match(pattern) or fnmatch.fnmatch(path_str, pattern):
        return True
    # Try matching without recursive prefix for root files
    if pattern.startswith("**/"):
        return fnmatch.fnmatch(path_str, pattern[3:])
    return False


# --8<-- [end:models]


# --- Knowledge Store ---


# --8<-- [start:store]
class KnowledgeStore:
    """
    Wraps NetworkX for graph operations and handles atomic JSON persistence.
    """

    def __init__(self, knowledge_map: KnowledgeMap | None = None) -> None:
        self._graph = nx.DiGraph()
        self._entity_lookup: dict[str, SemanticEntity] = {}
        self.root_path = "."
        self.last_updated = datetime.now()
        self.frontier_state: ExplorationFrontier | None = None

        if knowledge_map:
            self._load_from_map(knowledge_map)

    def _load_from_map(self, km: KnowledgeMap) -> None:
        """Load a KnowledgeMap into the graph."""
        self.root_path = km.root_path
        self.last_updated = km.last_updated
        self.frontier_state = km.frontier_state

        for entity in km.entities:
            self.add_entity(entity)

        for link in km.links:
            self.add_link(link)

    def add_entity(self, entity: SemanticEntity) -> None:
        """Add an entity to the graph."""
        self._graph.add_node(entity.id, **entity.model_dump())
        self._entity_lookup[entity.id] = entity

    def add_link(self, link: SemanticLink) -> None:
        """Add a relationship to the graph."""
        self._graph.add_edge(
            link.source_id,
            link.target_id,
            relationship=link.relationship,
            weight=link.weight,
        )

    def find_orphans(self) -> list[SemanticEntity]:
        """Find entities with no connections."""
        if not self._graph.nodes():
            return []
        orphan_ids = [
            n for n in self._graph.nodes() if self._graph.degree(n) == 0
        ]
        return [
            self._entity_lookup[eid]
            for eid in orphan_ids
            if eid in self._entity_lookup
        ]

    def find_central_entities(self, top_n: int = 10) -> list[SemanticEntity]:
        """Find most central entities using Degree Centrality."""
        if not self._graph.nodes():
            return []
        try:
            centrality = nx.degree_centrality(self._graph)
            sorted_ids = sorted(
                centrality.keys(), key=lambda x: centrality[x], reverse=True
            )
            return [
                self._entity_lookup[eid]
                for eid in sorted_ids[:top_n]
                if eid in self._entity_lookup
            ]
        except Exception:
            return []

    def to_knowledge_map(self) -> KnowledgeMap:
        """Export to KnowledgeMap."""
        entities = list(self._entity_lookup.values())
        links = []
        for u, v, data in self._graph.edges(data=True):
            links.append(
                SemanticLink(
                    source_id=u,
                    target_id=v,
                    relationship=data.get("relationship", "references"),
                    weight=data.get("weight", 1.0),
                )
            )
        return KnowledgeMap(
            entities=entities,
            links=links,
            root_path=self.root_path,
            last_updated=datetime.now(),
            frontier_state=self.frontier_state,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        km = self.to_knowledge_map()
        json_data = km.model_dump_json(indent=2)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json_data, encoding="utf-8")
        temp_path.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeStore:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"KnowledgeMap not found at {path}")
        json_data = path.read_text(encoding="utf-8")
        km = KnowledgeMap.model_validate_json(json_data)
        return cls(km)


# --8<-- [end:store]


# --- Extraction Models ---


class ExtractionResult(BaseModel):
    """Result of entity extraction from a file."""

    entities: list[SemanticEntity]
    links: list[SemanticLink]


class LLMExtractionResult(BaseModel):
    """Result of LLM extraction including token usage."""

    entities: list[SemanticEntity] = Field(default_factory=list)
    links: list[SemanticLink] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


@dataclass
class ExtractionDeps:
    """Dependencies for the extraction agent."""

    file_path: str
    content: str


# --- AST Extraction ---


# --8<-- [start:extraction]
class ASTExtractor:
    """Extracts structural entities using Python's AST."""

    def extract(self, file_path: str, content: str) -> ExtractionResult:
        entities: list[SemanticEntity] = []
        links: list[SemanticLink] = []
        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError:
            return ExtractionResult(entities=[], links=[])

        scope_name = self._path_to_scope(file_path)
        module_id = generate_entity_id("module", scope_name, "__init__")
        entities.append(
            SemanticEntity(
                id=module_id,
                name=Path(file_path).name,
                entity_type="module",
                summary=f"Module {scope_name}",
                location=file_path,
            )
        )

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_id = generate_entity_id("class", scope_name, node.name)
                entities.append(
                    SemanticEntity(
                        id=class_id,
                        name=node.name,
                        entity_type="class",
                        summary=self._get_docstring(node)
                        or f"Class {node.name}",
                        location=file_path,
                    )
                )
                links.append(
                    SemanticLink(
                        source_id=module_id,
                        target_id=class_id,
                        relationship="defines",
                    )
                )
            elif isinstance(node, ast.FunctionDef):
                func_id = generate_entity_id("function", scope_name, node.name)
                entities.append(
                    SemanticEntity(
                        id=func_id,
                        name=node.name,
                        entity_type="function",
                        summary=self._get_docstring(node)
                        or f"Function {node.name}",
                        location=file_path,
                    )
                )
                links.append(
                    SemanticLink(
                        source_id=module_id,
                        target_id=func_id,
                        relationship="defines",
                    )
                )
        return ExtractionResult(entities=entities, links=links)

    def _path_to_scope(self, file_path: str) -> str:
        p = Path(file_path)
        parts = list(p.parts)
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        try:
            src_idx = parts.index("src")
            return ".".join(parts[src_idx + 1 :])
        except ValueError:
            return ".".join(parts)

    def _get_docstring(self, node: Any) -> str | None:
        return ast.get_docstring(node)


# --- Markdown Extraction ---

# Regex for --8<-- snippet markers (not in marktripy AST)
SNIPPET_RE = re.compile(r'--8<--\s*"([^"]+)"')


class MarkdownExtractor:
    """Extracts structural entities from Markdown files using marktripy."""

    def __init__(self) -> None:
        self._parser = MarkdownItParser()

    def extract(self, file_path: str, content: str) -> ExtractionResult:
        """Extract structure from a markdown file using AST parsing."""
        entities: list[SemanticEntity] = []
        links: list[SemanticLink] = []

        scope_name = self._path_to_scope(file_path)
        doc_id = generate_entity_id("document", scope_name, "__doc__")

        # Parse markdown to AST
        ast = self._parser.parse(content)

        # Extract title from first heading or filename
        title = self._extract_title(ast) or Path(file_path).stem
        entities.append(
            SemanticEntity(
                id=doc_id,
                name=title,
                entity_type="document",
                summary=f"Documentation: {title}",
                location=file_path,
            )
        )

        # Walk AST to extract structure
        diagram_count = 0
        for node in ast.walk():
            if node.type == "heading":
                # Get heading text from children
                heading_text = self._get_text_content(node)
                level = node.attrs.get("level", 1) if node.attrs else 1
                section_id = generate_entity_id(
                    "section", scope_name, heading_text
                )
                entities.append(
                    SemanticEntity(
                        id=section_id,
                        name=heading_text,
                        entity_type="section",
                        summary=f"H{level}: {heading_text}",
                        location=file_path,
                        metadata={"level": level},
                    )
                )
                links.append(
                    SemanticLink(
                        source_id=doc_id,
                        target_id=section_id,
                        relationship="contains",
                    )
                )

            elif node.type == "code_block":
                block_content = node.content or ""
                lang = node.attrs.get("language", "") if node.attrs else ""

                # Check for mermaid diagrams
                if lang == "mermaid":
                    diagram_type = self._detect_diagram_type(block_content)
                    diagram_id = generate_entity_id(
                        "diagram", scope_name, f"mermaid_{diagram_count}"
                    )
                    diagram_count += 1
                    entities.append(
                        SemanticEntity(
                            id=diagram_id,
                            name=f"{diagram_type} diagram",
                            entity_type="diagram",
                            summary=f"Mermaid {diagram_type} diagram",
                            location=file_path,
                            metadata={"diagram_type": diagram_type},
                        )
                    )
                    links.append(
                        SemanticLink(
                            source_id=doc_id,
                            target_id=diagram_id,
                            relationship="contains",
                        )
                    )

                # Check for --8<-- snippet references in code blocks
                for match in SNIPPET_RE.finditer(block_content):
                    snippet_ref = match.group(1)
                    ref_id = generate_entity_id(
                        "code_ref", scope_name, snippet_ref
                    )
                    entities.append(
                        SemanticEntity(
                            id=ref_id,
                            name=snippet_ref,
                            entity_type="code_reference",
                            summary=f"Code snippet from {snippet_ref}",
                            location=file_path,
                            metadata={
                                "source_file": snippet_ref.split(":")[0]
                            },
                        )
                    )
                    links.append(
                        SemanticLink(
                            source_id=doc_id,
                            target_id=ref_id,
                            relationship="references",
                        )
                    )

        # Also check raw content for --8<-- outside code blocks
        for match in SNIPPET_RE.finditer(content):
            snippet_ref = match.group(1)
            ref_id = generate_entity_id("code_ref", scope_name, snippet_ref)
            # Avoid duplicates
            if not any(e.id == ref_id for e in entities):
                entities.append(
                    SemanticEntity(
                        id=ref_id,
                        name=snippet_ref,
                        entity_type="code_reference",
                        summary=f"Code snippet from {snippet_ref}",
                        location=file_path,
                        metadata={"source_file": snippet_ref.split(":")[0]},
                    )
                )
                links.append(
                    SemanticLink(
                        source_id=doc_id,
                        target_id=ref_id,
                        relationship="references",
                    )
                )

        return ExtractionResult(entities=entities, links=links)

    def _extract_title(self, ast: Any) -> str | None:
        """Extract first H1 heading as title."""
        for node in ast.walk():
            if node.type == "heading":
                level = node.attrs.get("level", 1) if node.attrs else 1
                if level == 1:
                    return self._get_text_content(node)
        return None

    def _get_text_content(self, node: Any) -> str:
        """Recursively get text content from a node."""
        if node.content:
            return node.content
        if hasattr(node, "children") and node.children:
            return "".join(
                self._get_text_content(child) for child in node.children
            )
        return ""

    def _path_to_scope(self, file_path: str) -> str:
        """Convert file path to scope identifier."""
        p = Path(file_path)
        parts = list(p.parts)
        if parts[-1].endswith(".md"):
            parts[-1] = parts[-1][:-3]
        for marker in ("docs", "dev", "src"):
            if marker in parts:
                idx = parts.index(marker)
                return ".".join(parts[idx:])
        return ".".join(parts[-3:])

    def _detect_diagram_type(self, content: str) -> str:
        """Detect mermaid diagram type from content."""
        first_line = content.split("\n")[0].strip().lower()
        if "flowchart" in first_line or "graph" in first_line:
            return "flowchart"
        if "sequence" in first_line:
            return "sequence"
        if "state" in first_line:
            return "state"
        if "class" in first_line:
            return "class"
        if "er" in first_line:
            return "er"
        return "diagram"


# --- LLM Extraction ---

EXTRACTOR_SYSTEM_PROMPT = (
    "You are a code analysis expert. "
    "Extract semantic concepts from source code.\n"
    "Analyze the provided code and identify high-level 'concepts'.\n"
    "Return a list of NEW entities found, or updates to existing ones."
)


def _extractor_context_prompt(ctx: RunContext[ExtractionDeps]) -> str:
    """Inject file context into the prompt."""
    return (
        f"\nAnalyzing file: {ctx.deps.file_path}\n"
        f"Content snippet:\n{ctx.deps.content[:4000]}"
    )


def create_extractor_agent(
    model: Model | None = None,
) -> Agent[ExtractionDeps, ExtractionResult]:
    """
    Create an extractor agent with optional model override.

    Args:
        model: pydantic-ai Model instance. If None, uses default.

    Returns:
        Configured extractor agent.
    """
    agent: Agent[ExtractionDeps, ExtractionResult] = Agent(
        model or get_model(),
        system_prompt=EXTRACTOR_SYSTEM_PROMPT,
        deps_type=ExtractionDeps,
        output_type=ExtractionResult,
    )
    agent.system_prompt(_extractor_context_prompt)
    return agent


# Default agent for backward compatibility
extractor_agent = create_extractor_agent()


class LLMExtractor:
    """Wraps the pydantic-ai Agent to provide semantic extraction."""

    def __init__(self, model: Model | None = None) -> None:
        """
        Initialize the LLM extractor.

        Args:
            model: Optional model override. If None, uses default.
        """
        self._agent = (
            create_extractor_agent(model) if model else extractor_agent
        )

    async def extract(
        self, file_path: str, content: str
    ) -> LLMExtractionResult:
        """
        Extract semantic entities from file content.

        Returns:
            LLMExtractionResult with entities, links, and token usage.
        """
        result = await self._agent.run(
            "Extract semantic entities and relationships.",
            deps=ExtractionDeps(file_path=file_path, content=content),
        )

        # Extract token usage from pydantic-ai result
        usage = result.usage()
        token_usage = TokenUsage(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
        )

        logfire.info(
            "LLM extraction complete",
            file=file_path,
            entities=len(result.data.entities),
            input_tokens=token_usage.input_tokens,
            output_tokens=token_usage.output_tokens,
        )

        return LLMExtractionResult(
            entities=result.data.entities,
            links=result.data.links,
            token_usage=token_usage,
        )


# --8<-- [end:extraction]


# --- Graph Nodes & State ---


# --8<-- [start:graph]
@dataclass
class CartographerState:
    """Runtime state for the exploration graph."""

    frontier: ExplorationFrontier = field(default_factory=ExplorationFrontier)
    store: KnowledgeStore | None = None
    current_depth: int = 0
    # Token accounting
    total_token_usage: TokenUsage = field(default_factory=TokenUsage)
    files_processed: int = 0
    llm_calls: int = 0


@dataclass
class CartographerDeps:
    """Dependencies for the Cartographer."""

    boundary: ExplorationBoundary = field(default_factory=ExplorationBoundary)
    storage_path: str | None = None
    model: Model | None = None


@dataclass
class ExploreNode(BaseNode[CartographerState, CartographerDeps, KnowledgeMap]):
    """Reconnaissance: List files/directories at current frontier."""

    async def run(
        self,
        ctx: GraphRunContext[CartographerState, CartographerDeps],
    ) -> ExtractNode | CompleteNode:
        frontier = ctx.state.frontier
        boundary = ctx.deps.boundary

        with logfire.span(
            "explore_node",
            pending_count=len(frontier.pending),
            explored_count=len(frontier.explored),
        ):
            if not frontier.pending:
                logfire.info("No pending paths, completing")
                return CompleteNode()

            batch_size = 5
            to_explore = frontier.pending[:batch_size]
            frontier.pending = frontier.pending[batch_size:]

            discovered_files: list[str] = []

            # Get root path for relative matching (with null check)
            root_path = "."
            if ctx.state.store is not None:
                root_path = ctx.state.store.root_path

        for path in to_explore:
            if path in frontier.explored:
                continue

            # Check depth
            depth = frontier.depth_map.get(path, 0)
            if depth > boundary.max_depth:
                continue

            frontier.explored.add(path)
            p = Path(path)
            if not p.exists():
                continue

            # Discovery Heartbeat
            if p.is_dir():
                print(f"   [Scout] Entering directory: {p.name}/")
                try:
                    for child in p.iterdir():
                        child_str = str(child)
                        # Compute relative path for matching
                        try:
                            rel_path = child.relative_to(
                                Path(root_path).resolve()
                            )
                        except ValueError:
                            rel_path = child

                        # Filter child before adding
                        exclude = any(
                            _is_match(rel_path, pattern)
                            for pattern in boundary.exclude_patterns
                        )
                        if exclude:
                            continue

                        if child.is_dir():
                            if child_str not in frontier.explored:
                                frontier.pending.append(child_str)
                                frontier.depth_map[child_str] = depth + 1
                        elif child.is_file():
                            include = any(
                                _is_match(rel_path, pattern)
                                for pattern in boundary.include_patterns
                            )
                            if include:
                                discovered_files.append(child_str)
                except PermissionError:
                    logfire.warn("Permission denied", path=path)
            elif p.is_file():
                discovered_files.append(path)

            logfire.info(
                "Exploration batch complete",
                discovered_files=len(discovered_files),
                pending_remaining=len(frontier.pending),
            )

            if not discovered_files and not frontier.pending:
                return CompleteNode()

            if not discovered_files:
                return ExploreNode()

            return ExtractNode(files_to_process=discovered_files)


@dataclass
class ExtractNode(BaseNode[CartographerState, CartographerDeps, KnowledgeMap]):
    """Entity Extraction: Read files and identify semantic entities."""

    files_to_process: list[str]

    async def run(
        self,
        ctx: GraphRunContext[CartographerState, CartographerDeps],
    ) -> MapNode:
        boundary = ctx.deps.boundary
        extracted_entities: list[SemanticEntity] = []
        extracted_links: list[SemanticLink] = []
        ast_extractor = ASTExtractor()
        md_extractor = MarkdownExtractor()
        llm_extractor = LLMExtractor(model=ctx.deps.model)

        # Token accounting for this batch
        batch_token_usage = TokenUsage()
        batch_llm_calls = 0
        batch_files_processed = 0

        with logfire.span(
            "extract_node",
            file_count=len(self.files_to_process),
            dry_run=boundary.dry_run,
        ):
            for file_path in self.files_to_process:
                try:
                    content = Path(file_path).read_text(
                        encoding="utf-8", errors="ignore"
                    )
                except Exception as e:
                    logfire.warn(
                        "Failed to read file", path=file_path, error=str(e)
                    )
                    continue

                batch_files_processed += 1
                file_name = Path(file_path).name

                # Choose extractor based on file type
                if file_path.endswith(".py"):
                    print(f"   [Analyzer] AST: {file_name}")
                    result = ast_extractor.extract(file_path, content)
                elif file_path.endswith(".md"):
                    print(f"   [Analyzer] MD:  {file_name}")
                    result = md_extractor.extract(file_path, content)
                else:
                    # Skip unsupported file types
                    continue

                extracted_entities.extend(result.entities)
                extracted_links.extend(result.links)

                if not boundary.dry_run:
                    try:
                        print(f"   [Brain] Semantic: {file_name}")
                        llm_result = await llm_extractor.extract(
                            file_path, content
                        )
                        extracted_entities.extend(llm_result.entities)
                        extracted_links.extend(llm_result.links)
                        # Accumulate token usage
                        batch_token_usage = (
                            batch_token_usage + llm_result.token_usage
                        )
                        batch_llm_calls += 1
                    except Exception as e:
                        logfire.warn(
                            "LLM extraction failed",
                            path=file_path,
                            error=str(e),
                        )

            # Update state with token accounting
            ctx.state.total_token_usage = (
                ctx.state.total_token_usage + batch_token_usage
            )
            ctx.state.files_processed += batch_files_processed
            ctx.state.llm_calls += batch_llm_calls

            logfire.info(
                "Extraction complete",
                entities_found=len(extracted_entities),
                links_found=len(extracted_links),
                batch_input_tokens=batch_token_usage.input_tokens,
                batch_output_tokens=batch_token_usage.output_tokens,
                total_input_tokens=ctx.state.total_token_usage.input_tokens,
                total_output_tokens=ctx.state.total_token_usage.output_tokens,
            )

        return MapNode(
            new_entities=extracted_entities, new_links=extracted_links
        )


@dataclass
class MapNode(BaseNode[CartographerState, CartographerDeps, KnowledgeMap]):
    """Relationship Mapping: Update graph and persist."""

    new_entities: list[SemanticEntity]
    new_links: list[SemanticLink]

    async def run(
        self,
        ctx: GraphRunContext[CartographerState, CartographerDeps],
    ) -> ExploreNode | CompleteNode:
        with logfire.span(
            "map_node",
            new_entities=len(self.new_entities),
            new_links=len(self.new_links),
        ):
            if ctx.state.store is None:
                ctx.state.store = KnowledgeStore()
            store = ctx.state.store

            for entity in self.new_entities:
                store.add_entity(entity)
            for link in self.new_links:
                store.add_link(link)

            store.frontier_state = ctx.state.frontier
            km = store.to_knowledge_map()
            modules = [e for e in km.entities if e.entity_type == "module"]

            logfire.info(
                "Graph updated",
                total_entities=len(km.entities),
                total_links=len(km.links),
                modules_found=len(modules),
            )

            if len(modules) >= ctx.deps.boundary.max_files:
                logfire.info("Max files reached, completing")
                return CompleteNode()

            if ctx.state.frontier.pending:
                return ExploreNode()
            return CompleteNode()


@dataclass
class CompleteNode(
    BaseNode[CartographerState, CartographerDeps, KnowledgeMap]
):
    """Finalize exploration and return the knowledge map."""

    async def run(
        self,
        ctx: GraphRunContext[CartographerState, CartographerDeps],
    ) -> End[KnowledgeMap]:
        with logfire.span("complete_node"):
            if ctx.state.store is None:
                ctx.state.store = KnowledgeStore()
            store = ctx.state.store
            store.frontier_state = ctx.state.frontier

            km = store.to_knowledge_map()

            # Add token accounting to the result
            km.token_usage = ctx.state.total_token_usage
            km.files_processed = ctx.state.files_processed
            km.llm_calls = ctx.state.llm_calls

            logfire.info(
                "Exploration complete",
                total_entities=len(km.entities),
                total_links=len(km.links),
                explored_paths=len(ctx.state.frontier.explored),
                files_processed=km.files_processed,
                llm_calls=km.llm_calls,
                input_tokens=km.token_usage.input_tokens,
                output_tokens=km.token_usage.output_tokens,
                total_tokens=km.token_usage.total_tokens,
            )

            if ctx.deps.storage_path:
                store.save(ctx.deps.storage_path)
                logfire.info("Knowledge map saved", path=ctx.deps.storage_path)

            return End(km)


# --8<-- [end:graph]


# --- Entry Point ---


# --8<-- [start:entry]
async def explore_domain(
    root_path: str,
    boundary: ExplorationBoundary | None = None,
    storage_path: str | None = None,
    *,
    model: Model | None = None,
) -> KnowledgeMap:
    """
    Explore a domain and build a knowledge map.

    Args:
        root_path: The root directory to explore.
        boundary: Exploration boundaries (depth, file limits, patterns).
        storage_path: Path to save/resume the knowledge map.
        model: Optional pydantic-ai Model for LLM extraction.
            If None, uses default model.

    Returns:
        KnowledgeMap containing discovered entities and relationships.
    """
    if boundary is None:
        boundary = ExplorationBoundary()

    with logfire.span(
        "explore_domain",
        root_path=root_path,
        max_depth=boundary.max_depth,
        max_files=boundary.max_files,
        dry_run=boundary.dry_run,
    ):
        deps = CartographerDeps(
            boundary=boundary, storage_path=storage_path, model=model
        )
        abs_root = str(Path(root_path).resolve())
        frontier = ExplorationFrontier(
            pending=[abs_root], depth_map={abs_root: 0}
        )
        store = KnowledgeStore()
        store.root_path = root_path

        if storage_path and Path(storage_path).exists():
            try:
                store = KnowledgeStore.load(storage_path)
                if store.frontier_state:
                    frontier = store.frontier_state
                    logfire.info(
                        "Resumed from existing map",
                        explored=len(frontier.explored),
                        pending=len(frontier.pending),
                    )
            except Exception as e:
                logfire.warn("Failed to load existing map", error=str(e))

        state = CartographerState(frontier=frontier, store=store)
        graph = Graph(nodes=[ExploreNode, ExtractNode, MapNode, CompleteNode])
        result = await graph.run(ExploreNode(), state=state, deps=deps)
        return result.output


# --8<-- [end:entry]


# --8<-- [start:main]
if __name__ == "__main__":
    import asyncio
    import sys

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree

    def render_results(km: KnowledgeMap, console: Console) -> None:
        """Render exploration results with rich visualization."""
        console.print()
        console.rule("[bold blue]Domain Exploration Results")
        console.print()

        # Summary panel
        console.print(
            Panel(
                f"[bold]Root:[/bold] {km.root_path}\n"
                f"[bold]Entities:[/bold] {len(km.entities)}\n"
                f"[bold]Links:[/bold] {len(km.links)}\n"
                f"[bold]Files Processed:[/bold] {km.files_processed}\n"
                f"[bold]Updated:[/bold] {km.last_updated.isoformat()}",
                title="Knowledge Map",
                border_style="blue",
            )
        )

        # Token accounting panel (only show if LLM was used)
        if km.llm_calls > 0:
            token_table = Table(
                title="[bold magenta]Token Accounting[/bold]",
                show_header=True,
                header_style="bold magenta",
            )
            token_table.add_column("Metric", width=20)
            token_table.add_column("Value", width=15, justify="right")

            token_table.add_row("LLM Calls", str(km.llm_calls))
            token_table.add_row(
                "Input Tokens", f"{km.token_usage.input_tokens:,}"
            )
            token_table.add_row(
                "Output Tokens", f"{km.token_usage.output_tokens:,}"
            )
            token_table.add_row(
                "[bold]Total Tokens[/bold]",
                f"[bold]{km.token_usage.total_tokens:,}[/bold]",
            )
            if km.llm_calls > 0:
                avg = km.token_usage.total_tokens // km.llm_calls
                token_table.add_row("Avg Tokens/Call", f"{avg:,}")

            console.print(token_table)
            console.print()
        else:
            console.print("[dim]Dry run mode - no LLM calls made[/dim]")
            console.print()

        # Entity breakdown by type
        type_counts: dict[str, int] = {}
        for e in km.entities:
            type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1

        type_table = Table(
            title="[bold]Entity Types[/bold]",
            show_header=True,
            header_style="bold cyan",
        )
        type_table.add_column("Type", width=15)
        type_table.add_column("Count", width=10, justify="right")

        for etype, count in sorted(
            type_counts.items(), key=lambda x: x[1], reverse=True
        ):
            type_table.add_row(etype, str(count))

        console.print(type_table)
        console.print()

        # Build tree visualization by location
        tree = Tree(
            f"[bold cyan]{km.root_path}[/]",
            guide_style="dim",
        )

        # Group entities by location
        by_location: dict[str, list[SemanticEntity]] = {}
        for e in km.entities:
            loc = e.location
            if loc not in by_location:
                by_location[loc] = []
            by_location[loc].append(e)

        for loc in sorted(by_location.keys())[:10]:  # Limit display
            entities = by_location[loc]
            loc_branch = tree.add(f"[dim]{Path(loc).name}[/]")
            for e in entities[:5]:  # Limit per file
                icon = {
                    "module": "[blue]M[/]",
                    "class": "[green]C[/]",
                    "function": "[yellow]F[/]",
                    "concept": "[magenta]~[/]",
                }.get(e.entity_type, "[dim]?[/]")
                loc_branch.add(f"{icon} {e.name}")
            if len(entities) > 5:
                loc_branch.add(f"[dim]... +{len(entities) - 5} more[/]")

        if len(by_location) > 10:
            tree.add(f"[dim]... +{len(by_location) - 10} more files[/]")

        console.print(tree)
        console.print()

        # Link statistics
        link_types: dict[str, int] = {}
        for link in km.links:
            link_types[link.relationship] = (
                link_types.get(link.relationship, 0) + 1
            )

        if link_types:
            link_table = Table(
                title="[bold]Relationships[/bold]",
                show_header=True,
                header_style="bold magenta",
            )
            link_table.add_column("Type", width=15)
            link_table.add_column("Count", width=10, justify="right")

            for ltype, count in sorted(
                link_types.items(), key=lambda x: x[1], reverse=True
            ):
                link_table.add_row(ltype, str(count))

            console.print(link_table)

    async def main() -> None:
        console = Console()
        console.print()
        console.rule("[bold]DEMO: The Cartographer - Domain Exploration")
        console.print()

        # Determine target path
        if len(sys.argv) > 1:
            target = sys.argv[1]
        else:
            # Default: explore the agentic_patterns source
            target = str(Path(__file__).parent)

        console.print(f"[dim]Exploring: {target}[/dim]")
        console.print("[dim]Mode: Dry run (AST only, no LLM calls)[/dim]")
        console.print()

        # Configure for demo
        boundary = ExplorationBoundary(
            max_depth=3,
            max_files=15,
            dry_run=True,  # AST only for fast demo
            include_patterns=["**/*.py"],
        )

        console.print(
            f"[dim]Config: depth={boundary.max_depth}, "
            f"max_files={boundary.max_files}[/dim]\n"
        )

        # Run exploration
        km = await explore_domain(root_path=target, boundary=boundary)

        # Display results
        render_results(km, console)

        # Graph analysis
        store = KnowledgeStore(km)
        central = store.find_central_entities(5)
        orphans = store.find_orphans()

        if central:
            console.print()
            console.print("[bold]Most Connected Entities:[/bold]")
            for e in central:
                console.print(f"  [green]{e.name}[/] ({e.entity_type})")

        if orphans:
            console.print()
            console.print(f"[dim]Orphan entities: {len(orphans)}[/]")

    asyncio.run(main())
# --8<-- [end:main]
