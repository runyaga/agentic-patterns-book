from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from agentic_patterns.domain_exploration import ASTExtractor
from agentic_patterns.domain_exploration import ExtractionResult
from agentic_patterns.domain_exploration import LLMExtractor
from agentic_patterns.domain_exploration import SemanticEntity
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


def test_ast_extractor_structure():
    """AST Extractor should find classes and functions."""
    extractor = ASTExtractor()
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


def test_ast_extractor_scope_ids():
    """IDs should be scoped correctly."""
    extractor = ASTExtractor()
    result = extractor.extract("src/pkg/mod.py", SAMPLE_CODE)

    # Check that IDs are consistent
    # Expected scope: pkg.mod
    class_entity = next(e for e in result.entities if e.name == "MyClass")
    # We verify the ID generation logic inside AST matches
    from agentic_patterns.domain_exploration import generate_entity_id

    expected_id = generate_entity_id("class", "pkg.mod", "MyClass")
    assert class_entity.id == expected_id


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
    mock_agent_result.data = mock_result
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
