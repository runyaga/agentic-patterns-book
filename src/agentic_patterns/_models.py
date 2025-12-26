"""Shared model configuration for all patterns."""

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def get_model(
    model_name: str = "gpt-oss:20b",
    base_url: str = "http://localhost:11434/v1",
) -> OpenAIChatModel:
    """Get configured model for agents.

    Args:
        model_name: The model identifier (default: gpt-oss:20b)
        base_url: The Ollama API base URL

    Returns:
        Configured OpenAIChatModel for use with pydantic-ai agents.
    """
    return OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(
            base_url=base_url,
            api_key="ollama",
        ),
    )
