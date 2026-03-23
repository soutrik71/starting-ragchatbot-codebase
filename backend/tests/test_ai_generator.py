"""
Unit tests for ai_generator.py

Scope: AIGenerator.generate_response() and _handle_tool_execution().
Strategy: patch `ai_generator.anthropic.Anthropic` so no real API calls are made.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from ai_generator import AIGenerator
from tests.conftest import make_direct_response, make_tool_use_response

FAKE_API_KEY = "sk-ant-test-key"
FAKE_MODEL = "claude-test-model"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client(mocker):
    """Patch anthropic.Anthropic and return the mock client instance."""
    client = MagicMock()
    mocker.patch("ai_generator.anthropic.Anthropic", return_value=client)
    return client


@pytest.fixture
def generator(mock_client):
    return AIGenerator(api_key=FAKE_API_KEY, model=FAKE_MODEL)


# ===========================================================================
# Direct response (no tool use)
# ===========================================================================

class TestGenerateResponseDirect:

    def test_returns_response_text_directly(self, generator, mock_client):
        mock_client.messages.create.return_value = make_direct_response("Forty-two.")
        result = generator.generate_response(query="What is the answer?")
        assert result == "Forty-two."

    def test_no_history_passes_only_system_prompt(self, generator, mock_client):
        mock_client.messages.create.return_value = make_direct_response("ok")
        generator.generate_response(query="Hello", conversation_history=None)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == AIGenerator.SYSTEM_PROMPT

    def test_with_history_appends_to_system_prompt(self, generator, mock_client):
        mock_client.messages.create.return_value = make_direct_response("ok")
        history = "User: Hi\nAssistant: Hello!"
        generator.generate_response(query="Hello again", conversation_history=history)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert history in call_kwargs["system"]
        assert AIGenerator.SYSTEM_PROMPT in call_kwargs["system"]

    def test_no_tools_omits_tool_choice_from_params(self, generator, mock_client):
        mock_client.messages.create.return_value = make_direct_response("ok")
        generator.generate_response(query="Hello", tools=None)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tool_choice" not in call_kwargs

    def test_with_tools_sets_tool_choice_auto(self, generator, mock_client):
        mock_client.messages.create.return_value = make_direct_response("ok")
        fake_tools = [{"name": "search", "description": "search tool"}]
        generator.generate_response(query="Search this", tools=fake_tools)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs.get("tool_choice") == {"type": "auto"}

    def test_user_query_is_sent_as_user_message(self, generator, mock_client):
        mock_client.messages.create.return_value = make_direct_response("ok")
        generator.generate_response(query="Explain RAG")

        call_kwargs = mock_client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "user"
        assert "Explain RAG" in messages[0]["content"]

    def test_base_params_use_correct_model_and_temperature(self, generator, mock_client):
        mock_client.messages.create.return_value = make_direct_response("ok")
        generator.generate_response(query="Hi")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == FAKE_MODEL
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 800


# ===========================================================================
# Tool use path
# ===========================================================================

class TestHandleToolExecution:

    def test_tool_use_triggers_second_api_call(self, generator, mock_client):
        """When the first response requests a tool, a second API call must follow."""
        tool_response = make_tool_use_response(
            "search_course_content", {"query": "python"}, tool_id="t1"
        )
        final_response = make_direct_response("Python is great.")
        mock_client.messages.create.side_effect = [tool_response, final_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Search results here"

        result = generator.generate_response(
            query="Tell me about Python",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert mock_client.messages.create.call_count == 2
        assert result == "Python is great."

    def test_tool_manager_is_called_with_correct_name_and_input(self, generator, mock_client):
        tool_response = make_tool_use_response(
            "search_course_content", {"query": "loops", "course_name": "Python"}, tool_id="t2"
        )
        mock_client.messages.create.side_effect = [
            tool_response,
            make_direct_response("Loops explanation."),
        ]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "loops content"

        generator.generate_response(
            query="Explain loops",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="loops", course_name="Python"
        )

    def test_second_api_call_message_structure(self, generator, mock_client):
        """Messages sent to the second call: [user_query, assistant_tool_use, user_tool_result]."""
        tool_response = make_tool_use_response(
            "search_course_content", {"query": "variables"}, tool_id="t3"
        )
        mock_client.messages.create.side_effect = [
            tool_response,
            make_direct_response("Variables answer."),
        ]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "variable results"

        generator.generate_response(
            query="Explain variables",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        assert messages[0]["role"] == "user"       # original query
        assert messages[1]["role"] == "assistant"  # tool use request
        assert messages[2]["role"] == "user"       # tool result
        # The tool result message should contain the tool_result type
        tool_result_content = messages[2]["content"]
        assert isinstance(tool_result_content, list)
        assert tool_result_content[0]["type"] == "tool_result"
        assert tool_result_content[0]["tool_use_id"] == "t3"

    def test_final_api_call_does_not_include_tools(self, generator, mock_client):
        """The second (final) API call must NOT include the tools parameter."""
        tool_response = make_tool_use_response(
            "search_course_content", {"query": "data types"}, tool_id="t4"
        )
        mock_client.messages.create.side_effect = [
            tool_response,
            make_direct_response("Data types answer."),
        ]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "data type results"

        generator.generate_response(
            query="Explain data types",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert "tools" not in second_call_kwargs
