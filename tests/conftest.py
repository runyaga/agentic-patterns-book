"""Shared pytest fixtures for all tests."""

import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


@pytest.fixture
def ollama_model():
    """Create Ollama model for testing (when not mocked)."""
    return OpenAIChatModel(
        model_name="gpt-oss:20b",
        provider=OpenAIProvider(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        ),
    )


@pytest.fixture
def model_config():
    """Default model configuration dict."""
    return {
        "model_name": "gpt-oss:20b",
        "base_url": "http://localhost:11434/v1",
    }
