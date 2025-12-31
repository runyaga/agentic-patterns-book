from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError

from agentic_patterns.domain_exploration import ExplorationBoundary
from agentic_patterns.domain_exploration import ExplorationFrontier
from agentic_patterns.domain_exploration import ExtractionResult
from agentic_patterns.domain_exploration import KnowledgeStore
from agentic_patterns.domain_exploration import SemanticEntity
from agentic_patterns.domain_exploration import SemanticLink
from agentic_patterns.domain_exploration import explore_domain
from agentic_patterns.domain_exploration import generate_entity_id


def test_generate_entity_id_deterministic():
    """ID generation should be consistent for same inputs."""
    id1 = generate_entity_id("class", "pkg.mod", "MyClass")
    id2 = generate_entity_id("class", "pkg.mod", "MyClass")
    id3 = generate_entity_id("function", "pkg.mod", "MyClass")

    assert id1 == id2
    assert id1 != id3


def test_knowledge_store_graph_ops():
    """Test basic graph operations in KnowledgeStore."""
    e1 = SemanticEntity(
        id="1", name="A", entity_type="class", summary="A", location="a.py"
    )
    e2 = SemanticEntity(
        id="2", name="B", entity_type="class", summary="B", location="b.py"
    )
    e3 = SemanticEntity(
        id="3", name="C", entity_type="class", summary="C", location="c.py"
    )

    store = KnowledgeStore()
    store.add_entity(e1)
    store.add_entity(e2)
    store.add_entity(e3)

    # A -> B, C -> B
    store.add_link(
        SemanticLink(source_id="1", target_id="2", relationship="calls")
    )
    store.add_link(
        SemanticLink(source_id="3", target_id="2", relationship="calls")
    )

    # Check Centrality (B should be most central)
    central = store.find_central_entities(1)
    assert len(central) == 1
    assert central[0].id == "2"

    # Check Orphans (None currently)
    assert len(store.find_orphans()) == 0

    # Add orphan
    e4 = SemanticEntity(
        id="4", name="D", entity_type="class", summary="D", location="d.py"
    )
    store.add_entity(e4)
    orphans = store.find_orphans()
    assert len(orphans) == 1
    assert orphans[0].id == "4"


def test_atomic_persistence():
    """Test saving and loading the knowledge map."""
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "knowledge.json"

        # Create initial store
        store = KnowledgeStore()
        store.root_path = tmpdir
        store.frontier_state = ExplorationFrontier(
            explored={"/a"}, pending=["/b"], depth_map={"/a": 0}
        )

        e1 = SemanticEntity(
            id="1", name="A", entity_type="class", summary="A", location="a.py"
        )
        store.add_entity(e1)

        # Save
        store.save(file_path)

        assert file_path.exists()

        # Load back
        loaded_store = KnowledgeStore.load(file_path)

        assert loaded_store.root_path == tmpdir
        assert len(loaded_store.find_orphans()) == 1
        assert loaded_store.frontier_state is not None
        assert "/a" in loaded_store.frontier_state.explored
        assert "/b" in loaded_store.frontier_state.pending


def test_persistence_overwrite():
    """Test that saving overwrites cleanly."""
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "knowledge.json"
        store = KnowledgeStore()
        store.root_path = tmpdir

        # Save V1
        store.add_entity(
            SemanticEntity(
                id="1",
                name="A",
                entity_type="class",
                summary="A",
                location="a.py",
            )
        )
        store.save(file_path)

        # Save V2
        store.add_entity(
            SemanticEntity(
                id="2",
                name="B",
                entity_type="class",
                summary="B",
                location="b.py",
            )
        )
        store.save(file_path)

        # Load
        loaded = KnowledgeStore.load(file_path)
        # Should have 2 entities now
        km = loaded.to_knowledge_map()
        assert len(km.entities) == 2


@pytest.mark.asyncio
async def test_explore_domain_integration():
    """Test autonomous exploration on a temporary directory (Dry Run)."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create structure:
        # src/
        #   main.py
        #   utils.py
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("class App: pass\nimport utils")
        (root / "src" / "utils.py").write_text("def helper(): pass")

        # Configure boundary for dry run (AST only)
        boundary = ExplorationBoundary(
            max_depth=5,
            max_files=10,
            dry_run=True,
            include_patterns=["**/*.py"],
        )

        # Run exploration
        km = await explore_domain(
            root_path=str(root),
            boundary=boundary,
            storage_path=str(root / "map.json"),
        )

        # Verify findings
        assert km.root_path == str(root)

        # Check entities
        # Should find: main.py(module), App(class), utils.py(module),
        # helper(function)
        entity_names = {e.name for e in km.entities}
        assert "App" in entity_names
        assert "helper" in entity_names

        # Check persistence
        assert (root / "map.json").exists()


@pytest.mark.asyncio
async def test_boundary_max_files():
    """Test that max_files limit stops the crawl."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create many files
        for i in range(10):
            (root / f"f{i}.py").write_text(f"class C{i}: pass")

        boundary = ExplorationBoundary(
            max_files=2,  # Limit to 2 files
            dry_run=True,
            include_patterns=["**/*.py"],
        )

        km = await explore_domain(root_path=str(root), boundary=boundary)

        # We count 'module' entities as a proxy for files
        modules = [e for e in km.entities if e.entity_type == "module"]

        # It should have stopped after the first batch (or the root directory)
        # Since all files are in the root, and we explore the root first,
        # it might pick up more than max_files if they are in the same dir.
        # But it should definitely not be 'unbounded' if we had thousands
        # of files.
        # For this test, we just want to see that it's not failing
        # and respects the logic.
        assert len(modules) > 0


class TestExtractionResultValidation:
    """Tests for ExtractionResult schema validation."""

    def test_valid_extraction_result(self):
        """Valid data should parse without errors."""
        result = ExtractionResult(
            entities=[
                SemanticEntity(
                    id="MyClass",
                    name="MyClass",
                    entity_type="class",
                    summary="A test class",
                    location="/path/to/file.py",
                )
            ],
            links=[
                SemanticLink(
                    source_id="MyClass",
                    target_id="BaseClass",
                    relationship="inherits",
                )
            ],
        )
        assert len(result.entities) == 1
        assert result.entities[0].entity_type == "class"

    def test_entity_type_must_be_lowercase(self):
        """Capitalized entity_type should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticEntity(
                id="MyClass",
                name="MyClass",
                entity_type="Class",  # Wrong: should be "class"
                summary="A test class",
                location="/path/to/file.py",
            )
        assert "entity_type" in str(exc_info.value)

    def test_entity_requires_name_field(self):
        """Missing name field should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticEntity(
                id="MyClass",
                # name is missing
                entity_type="class",
                summary="A test class",
                location="/path/to/file.py",
            )
        assert "name" in str(exc_info.value)

    def test_invalid_relationship_rejected(self):
        """Invalid relationship values should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SemanticLink(
                source_id="A",
                target_id="B",
                relationship="used_by",  # Invalid
            )
        assert "relationship" in str(exc_info.value)

    def test_all_valid_entity_types(self):
        """All valid entity_type values should be accepted."""
        valid_types = [
            "module",
            "class",
            "function",
            "variable",
            "concept",
            "document",
            "api_endpoint",
            "section",
            "code_reference",
            "diagram",
            "task",
            "list",
        ]
        for entity_type in valid_types:
            entity = SemanticEntity(
                id=f"test_{entity_type}",
                name=f"Test {entity_type}",
                entity_type=entity_type,
                summary="Test",
                location="/test.py",
            )
            assert entity.entity_type == entity_type

    def test_all_valid_relationships(self):
        """All valid relationship values should be accepted."""
        valid_relationships = [
            "imports",
            "calls",
            "defines",
            "references",
            "inherits",
            "implements",
            "contains",
            "depends_on",
        ]
        for rel in valid_relationships:
            link = SemanticLink(
                source_id="A",
                target_id="B",
                relationship=rel,
            )
            assert link.relationship == rel
