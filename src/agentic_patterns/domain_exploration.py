from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal

import networkx as nx
from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai import RunContext
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


# --- LLM Extraction ---

extractor_agent = Agent(
    get_model(),
    system_prompt=(
        "You are a code analysis expert. "
        "Extract semantic concepts from source code.\n"
        "Analyze the provided code and identify high-level 'concepts'.\n"
        "Return a list of NEW entities found, or updates to existing ones."
    ),
    deps_type=ExtractionDeps,
    output_type=ExtractionResult,
)


@extractor_agent.system_prompt
def inject_file_context(ctx: RunContext[ExtractionDeps]) -> str:
    return (
        f"\nAnalyzing file: {ctx.deps.file_path}\n"
        f"Content snippet:\n{ctx.deps.content[:4000]}"
    )


class LLMExtractor:
    """Wraps the pydantic-ai Agent to provide semantic extraction."""

    async def extract(self, file_path: str, content: str) -> ExtractionResult:
        result = await extractor_agent.run(
            "Extract semantic entities and relationships.",
            deps=ExtractionDeps(file_path=file_path, content=content),
        )
        return result.data


# --8<-- [end:extraction]


# --- Graph Nodes & State ---


# --8<-- [start:graph]
@dataclass
class CartographerState:
    """Runtime state for the exploration graph."""

    frontier: ExplorationFrontier = field(default_factory=ExplorationFrontier)
    store: KnowledgeStore | None = None
    current_depth: int = 0


@dataclass
class CartographerDeps:
    """Dependencies for the Cartographer."""

    boundary: ExplorationBoundary = field(default_factory=ExplorationBoundary)
    storage_path: str | None = None


@dataclass
class ExploreNode(BaseNode[CartographerState, CartographerDeps, KnowledgeMap]):
    """Reconnaissance: List files/directories at current frontier."""

    async def run(
        self,
        ctx: GraphRunContext[CartographerState, CartographerDeps],
    ) -> ExtractNode | CompleteNode:
        frontier = ctx.state.frontier
        boundary = ctx.deps.boundary

        if not frontier.pending:
            return CompleteNode()

        batch_size = 5
        to_explore = frontier.pending[:batch_size]
        frontier.pending = frontier.pending[batch_size:]

        discovered_files: list[str] = []

        for path in to_explore:
            if path in frontier.explored:
                continue

            depth = frontier.depth_map.get(path, 0)
            if depth > boundary.max_depth:
                continue

            frontier.explored.add(path)
            p = Path(path)
            if not p.exists():
                continue

            if p.is_dir():
                try:
                    for child in p.iterdir():
                        child_str = str(child)
                        # Compute relative path for matching
                        try:
                            rel_path = child.relative_to(
                                Path(ctx.state.store.root_path).resolve()
                            )
                        except ValueError:
                            rel_path = child

                        import fnmatch

                        def _is_match(p: Path, pat: str) -> bool:
                            # Try standard match
                            if p.match(pat) or fnmatch.fnmatch(str(p), pat):
                                return True
                            # Try matching without recursive prefix
                            # for root files
                            if pat.startswith("**/"):
                                return fnmatch.fnmatch(str(p), pat[3:])
                            return False

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
                    pass
            elif p.is_file():
                discovered_files.append(path)

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
        llm_extractor = LLMExtractor()

        for file_path in self.files_to_process:
            try:
                content = Path(file_path).read_text(
                    encoding="utf-8", errors="ignore"
                )
            except Exception:
                continue

            ast_result = ast_extractor.extract(file_path, content)
            extracted_entities.extend(ast_result.entities)
            extracted_links.extend(ast_result.links)

            if not boundary.dry_run:
                try:
                    llm_result = await llm_extractor.extract(
                        file_path, content
                    )
                    extracted_entities.extend(llm_result.entities)
                    extracted_links.extend(llm_result.links)
                except Exception:
                    pass
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
        if ctx.state.store is None:
            ctx.state.store = KnowledgeStore()
        store = ctx.state.store

        for entity in self.new_entities:
            store.add_entity(entity)
        for link in self.new_links:
            store.add_link(link)

        store.frontier_state = ctx.state.frontier
        modules = [
            e
            for e in store.to_knowledge_map().entities
            if e.entity_type == "module"
        ]
        if len(modules) >= ctx.deps.boundary.max_files:
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
        if ctx.state.store is None:
            ctx.state.store = KnowledgeStore()
        store = ctx.state.store
        store.frontier_state = ctx.state.frontier
        if ctx.deps.storage_path:
            store.save(ctx.deps.storage_path)
        return End(store.to_knowledge_map())


# --8<-- [end:graph]


# --- Entry Point ---


# --8<-- [start:entry]
async def explore_domain(
    root_path: str,
    boundary: ExplorationBoundary | None = None,
    storage_path: str | None = None,
) -> KnowledgeMap:
    if boundary is None:
        boundary = ExplorationBoundary()
    deps = CartographerDeps(boundary=boundary, storage_path=storage_path)
    abs_root = str(Path(root_path).resolve())
    frontier = ExplorationFrontier(pending=[abs_root], depth_map={abs_root: 0})
    store = KnowledgeStore()
    store.root_path = root_path
    if storage_path and Path(storage_path).exists():
        try:
            store = KnowledgeStore.load(storage_path)
            if store.frontier_state:
                frontier = store.frontier_state
        except Exception:
            pass
    state = CartographerState(frontier=frontier, store=store)
    graph = Graph(nodes=[ExploreNode, ExtractNode, MapNode, CompleteNode])
    result = await graph.run(ExploreNode(), state=state, deps=deps)
    return result.output


# --8<-- [end:entry]
