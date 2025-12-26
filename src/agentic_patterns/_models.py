"""Shared model configuration for all patterns."""

import os

import logfire
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

DEFAULT_MODEL = os.environ.get("REQUIRED_MODEL", "gpt-oss:20b")
DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/v1"

# Configure logfire if token is set
if os.environ.get("LOGFIRE_TOKEN"):
    logfire.configure(
        scrubbing=False,
        send_to_logfire="if-token-present",
    )
    logfire.instrument_pydantic_ai()


def get_model(
    model_name: str | None = None,
    base_url: str | None = None,
) -> OpenAIChatModel:
    """Get configured model for agents.

    Args:
        model_name: Model identifier. Env: REQUIRED_MODEL. Default: gpt-oss:20b
        base_url: Ollama API URL. Env: OLLAMA_URL. Default: localhost:11434

    Returns:
        Configured OpenAIChatModel for use with pydantic-ai agents.
    """
    return OpenAIChatModel(
        model_name=model_name or DEFAULT_MODEL,
        provider=OpenAIProvider(
            base_url=base_url or DEFAULT_URL,
            api_key="ollama",
        ),
    )
