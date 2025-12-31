from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns.domain_exploration import ExtractionResult
from agentic_patterns.domain_exploration import LLMExtractor
from agentic_patterns.domain_exploration import MarkdownExtractor
from agentic_patterns.domain_exploration import SemanticEntity
from agentic_patterns.domain_exploration import TreeSitterExtractor
from agentic_patterns.domain_exploration import extractor_agent

SAMPLE_CODE = """
class MyClass:
    '''This is a class.'''
    def my_method(self):
        pass

def my_function():
    '''This is a function.'''
    return True
"""


def test_tree_sitter_extractor_structure():
    """TreeSitter Extractor should find classes and functions."""
    extractor = TreeSitterExtractor()
    result = extractor.extract("src/pkg/mod.py", SAMPLE_CODE)

    # Check entities
    assert len(result.entities) >= 3  # Module + Class + Function

    types = [e.entity_type for e in result.entities]
    assert "module" in types
    assert "class" in types
    assert "function" in types

    # Check names
    names = [e.name for e in result.entities]
    assert "mod.py" in names
    assert "MyClass" in names
    assert "my_function" in names

    # Check docstrings
    class_entity = next(e for e in result.entities if e.name == "MyClass")
    assert "This is a class" in class_entity.summary


def test_tree_sitter_extractor_scope_ids():
    """IDs should be scoped correctly."""
    extractor = TreeSitterExtractor()
    result = extractor.extract("src/pkg/mod.py", SAMPLE_CODE)

    # Check that IDs are consistent
    # Expected scope: pkg.mod
    class_entity = next(e for e in result.entities if e.name == "MyClass")
    from agentic_patterns.domain_exploration import generate_entity_id

    expected_id = generate_entity_id("class", "pkg.mod", "MyClass")
    assert class_entity.id == expected_id


def test_tree_sitter_extractor_inheritance():
    """TreeSitter Extractor should capture inheritance relationships."""
    code = """
class Parent:
    pass

class Child(Parent):
    pass
"""
    extractor = TreeSitterExtractor()
    result = extractor.extract("src/pkg/mod.py", code)

    # Check inheritance link exists
    inherits_links = [
        link for link in result.links if link.relationship == "inherits"
    ]
    assert len(inherits_links) == 1

    # Check Child has base_classes metadata
    child = next(e for e in result.entities if e.name == "Child")
    assert "Parent" in child.metadata.get("base_classes", [])


@pytest.mark.asyncio
async def test_llm_extractor_mock():
    """Test LLM extraction with a mock."""
    mock_result = ExtractionResult(
        entities=[
            SemanticEntity(
                id="123",
                name="Concept",
                entity_type="concept",
                summary="A cool concept",
                location="file.py",
            )
        ],
        links=[],
    )

    # Mock token usage
    mock_usage = MagicMock()
    mock_usage.input_tokens = 100
    mock_usage.output_tokens = 50

    # Mock the agent run result
    mock_agent_result = MagicMock()
    mock_agent_result.output = mock_result
    mock_agent_result.usage.return_value = mock_usage

    # Mock the agent run
    with patch.object(
        extractor_agent, "run", new_callable=AsyncMock
    ) as mock_run:
        mock_run.return_value = mock_agent_result

        extractor = LLMExtractor()
        result = await extractor.extract("file.py", "content")

        assert len(result.entities) == 1
        assert result.entities[0].name == "Concept"
        # Check token accounting
        assert result.token_usage.input_tokens == 100
        assert result.token_usage.output_tokens == 50
        assert result.token_usage.total_tokens == 150
        mock_run.assert_called_once()


SAMPLE_MARKDOWN_TASKS = """
# Project Tasks

## In Progress

- [ ] Implement feature A
- [x] Write tests for feature B
- [ ] Review PR #123
"""


def test_markdown_extractor_task_lists():
    """MarkdownExtractor should find task list items."""
    extractor = MarkdownExtractor()
    result = extractor.extract("docs/tasks.md", SAMPLE_MARKDOWN_TASKS)

    # Check entity types
    types = [e.entity_type for e in result.entities]
    assert "document" in types
    assert "section" in types
    assert "task" in types

    # Check tasks
    tasks = [e for e in result.entities if e.entity_type == "task"]
    assert len(tasks) == 3

    # Check task completion status
    done_tasks = [e for e in tasks if e.metadata.get("completed")]
    todo_tasks = [e for e in tasks if not e.metadata.get("completed")]
    assert len(done_tasks) == 1
    assert len(todo_tasks) == 2

    # Verify task names
    task_names = [e.name for e in tasks]
    assert "Implement feature A" in task_names
    assert "Write tests for feature B" in task_names


SAMPLE_MARKDOWN_LISTS = """
# Documentation

## Features

- Fast processing
- Easy to use
- Extensible API

## Installation Steps

1. Clone the repo
2. Install dependencies
3. Run setup
"""


def test_markdown_extractor_regular_lists():
    """MarkdownExtractor should find regular lists (bullet and ordered)."""
    extractor = MarkdownExtractor()
    result = extractor.extract("docs/readme.md", SAMPLE_MARKDOWN_LISTS)

    # Check for list entities
    lists = [e for e in result.entities if e.entity_type == "list"]
    assert len(lists) == 2

    # Check list types
    list_types = [e.metadata.get("list_type") for e in lists]
    assert "bullet" in list_types
    assert "ordered" in list_types

    # Check item counts
    bullet_list = next(
        e for e in lists if e.metadata.get("list_type") == "bullet"
    )
    ordered_list = next(
        e for e in lists if e.metadata.get("list_type") == "ordered"
    )
    assert bullet_list.metadata.get("item_count") == 3
    assert ordered_list.metadata.get("item_count") == 3

    # Check summary contains items
    assert "Fast processing" in bullet_list.summary
    assert "Clone the repo" in ordered_list.summary
