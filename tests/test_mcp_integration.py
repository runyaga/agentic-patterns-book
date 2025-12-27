"""Tests for the MCP Integration Pattern."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic_ai.mcp import MCPServerStdio

from agentic_patterns.mcp_integration import MCPDeps
from agentic_patterns.mcp_integration import create_calculator_server
from agentic_patterns.mcp_integration import create_filesystem_server
from agentic_patterns.mcp_integration import create_mcp_agent
from agentic_patterns.mcp_integration import process_tool_call_with_deps


class TestMCPDeps:
    """Test MCPDeps dataclass."""

    def test_deps_defaults(self):
        deps = MCPDeps()
        assert deps.user_id is None
        assert deps.request_id is None
        assert deps.context is None

    def test_deps_with_values(self):
        deps = MCPDeps(
            user_id="user123",
            request_id="req456",
            context={"key": "value"},
        )
        assert deps.user_id == "user123"
        assert deps.request_id == "req456"
        assert deps.context == {"key": "value"}


class TestServerCreation:
    """Test MCP server factory functions."""

    def test_create_calculator_server(self):
        server = create_calculator_server()
        assert server.command == sys.executable
        assert "-m" in server.args
        assert "agentic_patterns.mcp_servers.calculator" in server.args
        assert server.tool_prefix is None

    def test_create_calculator_server_with_prefix(self):
        server = create_calculator_server(tool_prefix="calc")
        assert server.tool_prefix == "calc"

    def test_create_filesystem_server(self):
        server = create_filesystem_server()
        assert server.command == sys.executable
        assert "-m" in server.args
        assert "agentic_patterns.mcp_servers.filesystem" in server.args

    def test_create_filesystem_server_with_prefix(self):
        server = create_filesystem_server(tool_prefix="fs")
        assert server.tool_prefix == "fs"


class TestAgentCreation:
    """Test agent creation functions."""

    def test_create_mcp_agent_default(self):
        agent = create_mcp_agent()
        assert agent is not None
        # Should have toolsets configured
        assert len(agent.toolsets) > 0

    def test_create_mcp_agent_custom_prompt(self):
        agent = create_mcp_agent(system_prompt="Custom prompt")
        assert "Custom prompt" in agent._system_prompts[0]

    def test_create_mcp_agent_multiple_servers(self):
        calc = create_calculator_server()
        fs = create_filesystem_server()
        agent = create_mcp_agent(servers=[calc, fs])
        # pydantic-ai adds internal toolsets, so check we have at least 2
        assert len(agent.toolsets) >= 2
        # Verify our MCP servers are in there
        mcp_servers = [
            t for t in agent.toolsets if isinstance(t, MCPServerStdio)
        ]
        assert len(mcp_servers) == 2

    def test_create_mcp_agent_with_deps(self):
        agent = create_mcp_agent(with_deps=True)
        assert agent._deps_type == MCPDeps


class TestProcessToolCallWithDeps:
    """Test deps propagation to MCP tools."""

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.deps = MCPDeps(
            user_id="user123",
            request_id="req456",
        )
        return ctx

    @pytest.fixture
    def mock_call_tool(self):
        return AsyncMock(return_value={"result": "ok"})

    @pytest.mark.asyncio
    async def test_deps_injected_into_meta(self, mock_ctx, mock_call_tool):
        """Deps should be passed as meta to call_tool."""
        await process_tool_call_with_deps(
            mock_ctx,
            mock_call_tool,
            "test_tool",
            {"arg": "value"},
        )

        mock_call_tool.assert_called_once()
        call_args = mock_call_tool.call_args
        assert call_args[0][0] == "test_tool"
        assert call_args[0][1] == {"arg": "value"}
        meta = call_args[0][2]
        assert meta["user_id"] == "user123"
        assert meta["request_id"] == "req456"

    @pytest.mark.asyncio
    async def test_empty_deps_no_meta(self, mock_call_tool):
        """Empty deps should pass None meta."""
        ctx = MagicMock()
        ctx.deps = MCPDeps()

        await process_tool_call_with_deps(
            ctx,
            mock_call_tool,
            "test_tool",
            {},
        )

        call_args = mock_call_tool.call_args
        # Meta should be None when all deps are None
        assert call_args[0][2] is None

    @pytest.mark.asyncio
    async def test_partial_deps(self, mock_call_tool):
        """Partial deps should only include non-None values."""
        ctx = MagicMock()
        ctx.deps = MCPDeps(user_id="user123")

        await process_tool_call_with_deps(
            ctx,
            mock_call_tool,
            "test_tool",
            {},
        )

        call_args = mock_call_tool.call_args
        meta = call_args[0][2]
        assert meta == {"user_id": "user123"}

    @pytest.mark.asyncio
    async def test_context_dict_passed(self, mock_call_tool):
        """Context dict should be passed in meta."""
        ctx = MagicMock()
        ctx.deps = MCPDeps(context={"custom": "data"})

        await process_tool_call_with_deps(
            ctx,
            mock_call_tool,
            "test_tool",
            {},
        )

        call_args = mock_call_tool.call_args
        meta = call_args[0][2]
        assert meta["context"] == {"custom": "data"}


class TestCalculatorServer:
    """Test calculator MCP server tools directly."""

    def test_add(self):
        from agentic_patterns.mcp_servers.calculator import add

        assert add(2, 3) == 5
        assert add(-1, 1) == 0
        assert add(1.5, 2.5) == 4.0

    def test_subtract(self):
        from agentic_patterns.mcp_servers.calculator import subtract

        assert subtract(5, 3) == 2
        assert subtract(0, 5) == -5

    def test_multiply(self):
        from agentic_patterns.mcp_servers.calculator import multiply

        assert multiply(3, 4) == 12
        assert multiply(-2, 3) == -6
        assert multiply(0, 100) == 0

    def test_divide(self):
        from agentic_patterns.mcp_servers.calculator import divide

        assert divide(10, 2) == 5
        assert divide(7, 2) == 3.5

    def test_divide_by_zero(self):
        from agentic_patterns.mcp_servers.calculator import divide

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10, 0)


class TestFilesystemServer:
    """Test filesystem MCP server tools directly."""

    @pytest.fixture
    def temp_dir(self):
        with TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "file1.txt").write_text("content1")
            (Path(tmpdir) / "file2.txt").write_text("content2")
            (Path(tmpdir) / "subdir").mkdir()
            yield tmpdir

    def test_list_directory(self, temp_dir):
        from agentic_patterns.mcp_servers.filesystem import list_directory

        files = list_directory(temp_dir)
        assert "file1.txt" in files
        assert "file2.txt" in files
        assert "subdir" in files

    def test_list_directory_not_exists(self):
        from agentic_patterns.mcp_servers.filesystem import list_directory

        with pytest.raises(ValueError, match="does not exist"):
            list_directory("/nonexistent/path")

    def test_list_directory_not_dir(self, temp_dir):
        from agentic_patterns.mcp_servers.filesystem import list_directory

        file_path = str(Path(temp_dir) / "file1.txt")
        with pytest.raises(ValueError, match="not a directory"):
            list_directory(file_path)

    def test_read_file(self, temp_dir):
        from agentic_patterns.mcp_servers.filesystem import read_file

        file_path = str(Path(temp_dir) / "file1.txt")
        content = read_file(file_path)
        assert content == "content1"

    def test_read_file_truncation(self, temp_dir):
        from agentic_patterns.mcp_servers.filesystem import read_file

        # Create a large file
        large_file = Path(temp_dir) / "large.txt"
        large_file.write_text("x" * 1000)

        content = read_file(str(large_file), max_chars=100)
        assert len(content) < 1000
        assert "truncated" in content

    def test_read_file_not_exists(self):
        from agentic_patterns.mcp_servers.filesystem import read_file

        with pytest.raises(ValueError, match="does not exist"):
            read_file("/nonexistent/file.txt")

    def test_file_exists(self, temp_dir):
        from agentic_patterns.mcp_servers.filesystem import file_exists

        assert file_exists(temp_dir) is True
        assert file_exists(str(Path(temp_dir) / "file1.txt")) is True
        assert file_exists("/nonexistent") is False

    def test_get_file_size(self, temp_dir):
        from agentic_patterns.mcp_servers.filesystem import get_file_size

        file_path = str(Path(temp_dir) / "file1.txt")
        size = get_file_size(file_path)
        assert size == len("content1")

    def test_get_file_size_not_exists(self):
        from agentic_patterns.mcp_servers.filesystem import get_file_size

        with pytest.raises(ValueError, match="does not exist"):
            get_file_size("/nonexistent")


class TestRunWithMCPTools:
    """Test the run_with_mcp_tools entry point."""

    @pytest.mark.asyncio
    async def test_run_with_mocked_agent(self):
        """Test run_with_mcp_tools with mocked agent."""
        mock_result = MagicMock()
        mock_result.output = "The answer is 42"

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "agentic_patterns.mcp_integration.create_mcp_agent",
            return_value=mock_agent,
        ):
            from agentic_patterns.mcp_integration import run_with_mcp_tools

            result = await run_with_mcp_tools("What is 6 * 7?")

        assert result == "The answer is 42"
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_deps(self):
        """Test run_with_mcp_tools passes deps correctly."""
        mock_result = MagicMock()
        mock_result.output = "Done"

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)

        deps = MCPDeps(user_id="test_user")

        with patch(
            "agentic_patterns.mcp_integration.create_mcp_agent",
            return_value=mock_agent,
        ) as mock_create:
            from agentic_patterns.mcp_integration import run_with_mcp_tools

            await run_with_mcp_tools("test", deps=deps)

        # Should have called create_mcp_agent with with_deps=True
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["with_deps"] is True

        # Should have passed deps to agent.run
        run_call_kwargs = mock_agent.run.call_args[1]
        assert run_call_kwargs["deps"] == deps
