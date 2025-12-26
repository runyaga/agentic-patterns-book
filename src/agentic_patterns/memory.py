"""
Memory Management Pattern Implementation (Idiomatic PydanticAI).

Based on the Agentic Design Patterns book Chapter 8:
Enable agents to maintain context and learn from interactions.
"""

import asyncio
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum

from pydantic import BaseModel
from pydantic import Field
from pydantic_ai import Agent, RunContext

from agentic_patterns._models import get_model


class MessageRole(str, Enum):
    """Role of a message sender."""

    USER = "user"
    AI = "ai"
    SYSTEM = "system"


class MemoryMessage(BaseModel):
    """A single message in memory."""

    role: MessageRole = Field(description="Who sent this message")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the message was created",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Optional metadata about the message",
    )


class ConversationSummary(BaseModel):
    """Summary of a conversation segment."""

    summary: str = Field(description="Condensed summary of the conversation")
    message_count: int = Field(description="Number of messages summarized")
    key_points: list[str] = Field(
        default_factory=list,
        description="Key points from the conversation",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the summary was created",
    )


class MemoryStats(BaseModel):
    """Statistics about memory usage."""

    total_messages: int = Field(description="Total messages stored")
    user_messages: int = Field(description="Number of user messages")
    ai_messages: int = Field(description="Number of AI messages")
    system_messages: int = Field(description="Number of system messages")
    approximate_tokens: int = Field(
        description="Approximate token count (chars/4)"
    )


@dataclass
class BufferMemory:
    """Stores complete history of all messages."""

    messages: list[MemoryMessage] = field(default_factory=list)
    max_messages: int | None = None

    def add_message(self, role: MessageRole, content: str):
        msg = MemoryMessage(role=role, content=content)
        self.messages.append(msg)
        if self.max_messages and len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_user_message(self, content: str):
        self.add_message(MessageRole.USER, content)

    def add_ai_message(self, content: str):
        self.add_message(MessageRole.AI, content)

    def get_context(self) -> str:
        return "\n".join(f"{m.role.value.upper()}: {m.content}" for m in self.messages)

    def get_stats(self) -> MemoryStats:
        user_count = sum(1 for m in self.messages if m.role == MessageRole.USER)
        ai_count = sum(1 for m in self.messages if m.role == MessageRole.AI)
        total_chars = sum(len(m.content) for m in self.messages)
        return MemoryStats(
            total_messages=len(self.messages),
            user_messages=user_count,
            ai_messages=ai_count,
            system_messages=0,
            approximate_tokens=total_chars // 4,
        )


@dataclass
class WindowMemory:
    """Only keeps the last N message exchanges."""

    messages: list[MemoryMessage] = field(default_factory=list)
    window_size: int = 10

    def add_message(self, role: MessageRole, content: str):
        self.messages.append(MemoryMessage(role=role, content=content))
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size :]

    def add_user_message(self, content: str):
        self.add_message(MessageRole.USER, content)

    def add_ai_message(self, content: str):
        self.add_message(MessageRole.AI, content)

    def get_context(self) -> str:
        return "\n".join(f"{m.role.value.upper()}: {m.content}" for m in self.messages)

    def get_stats(self) -> MemoryStats:
        user_count = sum(1 for m in self.messages if m.role == MessageRole.USER)
        ai_count = sum(1 for m in self.messages if m.role == MessageRole.AI)
        total_chars = sum(len(m.content) for m in self.messages)
        return MemoryStats(
            total_messages=len(self.messages),
            user_messages=user_count,
            ai_messages=ai_count,
            system_messages=0,
            approximate_tokens=total_chars // 4,
        )


@dataclass
class SummaryMemory:
    """Condenses older history into summaries."""

    messages: list[MemoryMessage] = field(default_factory=list)
    summaries: list[ConversationSummary] = field(default_factory=list)
    recent_window: int = 6
    summarize_threshold: int = 12

    def add_message(self, role: MessageRole, content: str):
        self.messages.append(MemoryMessage(role=role, content=content))

    def add_user_message(self, content: str):
        self.add_message(MessageRole.USER, content)

    def add_ai_message(self, content: str):
        self.add_message(MessageRole.AI, content)

    def get_context(self) -> str:
        parts = []
        if self.summaries:
            parts.append(f"Previous summary: {self.summaries[-1].summary}")
        recent = "\n".join(f"{m.role.value.upper()}: {m.content}" for m in self.messages)
        parts.append(f"Recent messages:\n{recent}")
        return "\n".join(parts)

    def get_stats(self) -> MemoryStats:
        user_count = sum(1 for m in self.messages if m.role == MessageRole.USER)
        ai_count = sum(1 for m in self.messages if m.role == MessageRole.AI)
        msg_chars = sum(len(m.content) for m in self.messages)
        sum_chars = sum(len(s.summary) for s in self.summaries)
        return MemoryStats(
            total_messages=len(self.messages),
            user_messages=user_count,
            ai_messages=ai_count,
            system_messages=0,
            approximate_tokens=(msg_chars + sum_chars) // 4,
        )


@dataclass
class MemoryDeps:
    """Dependencies for conversational agent."""

    memory: BufferMemory | WindowMemory | SummaryMemory


# Initialize model
model = get_model()

# Conversational agent with memory support
conversational_agent = Agent(
    model,
    system_prompt="You are a helpful assistant with conversation memory.",
    deps_type=MemoryDeps,
    output_type=str,
)


@conversational_agent.system_prompt
def add_memory_context(ctx: RunContext[MemoryDeps]) -> str:
    """Inject conversation history into the system prompt."""
    context = ctx.deps.memory.get_context()
    if not context:
        return "No previous conversation history."
    return f"Conversation history:\n{context}\n\nMaintain continuity with this history."


async def chat_with_memory(
    deps: MemoryDeps,
    user_input: str,
) -> str:
    """Process a user message with memory context via dependencies."""
    # Add user message to memory
    deps.memory.add_user_message(user_input)

    # Get response - context is injected automatically via @system_prompt
    result = await conversational_agent.run(user_input, deps=deps)
    response = result.output

    # Add AI response to memory
    deps.memory.add_ai_message(response)
    return response


async def run_conversation(
    memory_type: str = "buffer",
    window_size: int = 10,
) -> None:
    """Run an interactive conversation demo."""
    if memory_type == "window":
        memory = WindowMemory(window_size=window_size)
    elif memory_type == "summary":
        memory = SummaryMemory(recent_window=window_size // 2, summarize_threshold=window_size)
    else:
        memory = BufferMemory()

    deps = MemoryDeps(memory=memory)

    test_exchanges = [
        "Hello! My name is Alice and I'm interested in Python.",
        "What are the main benefits of using async/await?",
        "Can you give me a simple example?",
        "What was my name again?",
    ]

    print(f"\nUsing {type(memory).__name__}")
    print("=" * 60)
    for user_msg in test_exchanges:
        print(f"\nUSER: {user_msg}")
        response = await chat_with_memory(deps, user_msg)
        print(f"AI: {response}")

    stats = deps.memory.get_stats()
    print("\n" + "=" * 60)
    print(f"Stats: {stats.total_messages} messages, ~{stats.approximate_tokens} tokens")


if __name__ == "__main__":

    async def main() -> None:
        print("=" * 60)
        print("DEMO: Memory Management Pattern (Idiomatic)")
        print("=" * 60)

        await run_conversation(memory_type="buffer")
        print("\n\n")
        await run_conversation(memory_type="window", window_size=6)

    asyncio.run(main())