"""
Unit tests for search_tools.py

Scope: CourseSearchTool and ToolManager.
Strategy: inject a MagicMock VectorStore so no real ChromaDB is needed.
"""

import re
import pytest
from unittest.mock import MagicMock

from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(docs=None, metadata=None, lesson_link=None, course_link=None):
    """Return a mock VectorStore configured with canned search results."""
    store = MagicMock()
    results = SearchResults(
        documents=docs or ["Python is a high-level programming language."],
        metadata=metadata or [{"course_title": "Intro to Python", "lesson_number": 1, "chunk_index": 0}],
        distances=[0.25],
    )
    store.search.return_value = results
    store.get_lesson_link.return_value = lesson_link
    store.get_course_link.return_value = course_link
    return store


# ===========================================================================
# CourseSearchTool tests
# ===========================================================================

class TestCourseSearchToolExecute:

    def test_execute_returns_formatted_string(self):
        tool = CourseSearchTool(_make_store())
        result = tool.execute(query="python")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_result_contains_course_header(self):
        tool = CourseSearchTool(_make_store())
        result = tool.execute(query="python")
        # Formatted header looks like "[Intro to Python - Lesson 1]"
        assert "[Intro to Python" in result

    def test_execute_stores_markdown_link_in_last_sources_when_lesson_link_available(self):
        store = _make_store(lesson_link="https://example.com/lesson/1")
        tool = CourseSearchTool(store)
        tool.execute(query="python")
        assert len(tool.last_sources) == 1
        source = tool.last_sources[0]
        # Must match markdown link pattern [label](url)
        assert re.match(r"\[.+\]\(https://example\.com/lesson/1\)", source)

    def test_execute_falls_back_to_course_link_when_no_lesson_link(self):
        store = _make_store(lesson_link=None, course_link="https://example.com/course")
        tool = CourseSearchTool(store)
        tool.execute(query="python")
        source = tool.last_sources[0]
        assert "https://example.com/course" in source

    def test_execute_plain_text_source_when_no_links_available(self):
        store = _make_store(lesson_link=None, course_link=None)
        tool = CourseSearchTool(store)
        tool.execute(query="python")
        source = tool.last_sources[0]
        # No markdown link syntax — plain text only
        assert "](http" not in source
        assert "Intro to Python" in source

    def test_execute_handles_empty_results(self):
        # Use SearchResults with no error and no documents to hit the is_empty() branch
        store = MagicMock()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        tool = CourseSearchTool(store)
        result = tool.execute(query="obscure topic")
        assert "No relevant content found" in result

    def test_execute_handles_search_error(self):
        store = MagicMock()
        store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error="ChromaDB connection failed"
        )
        tool = CourseSearchTool(store)
        result = tool.execute(query="anything")
        assert "ChromaDB connection failed" in result

    def test_execute_with_no_lesson_number_skips_lesson_link_lookup(self):
        store = _make_store(
            metadata=[{"course_title": "Intro to Python", "lesson_number": None, "chunk_index": 0}],
            course_link="https://example.com/course",
        )
        tool = CourseSearchTool(store)
        tool.execute(query="python")
        # get_lesson_link should NOT have been called since lesson_number is None
        store.get_lesson_link.assert_not_called()
        # Falls back to course link
        source = tool.last_sources[0]
        assert "https://example.com/course" in source

    def test_execute_includes_filter_hint_in_empty_message_when_course_name_given(self):
        # No error, no documents → hits is_empty() branch which includes filter context
        store = MagicMock()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        tool = CourseSearchTool(store)
        result = tool.execute(query="loops", course_name="Intro to Python")
        assert "Intro to Python" in result


class TestCourseSearchToolDefinition:

    def test_get_tool_definition_has_correct_name(self):
        tool = CourseSearchTool(MagicMock())
        defn = tool.get_tool_definition()
        assert defn["name"] == "search_course_content"

    def test_get_tool_definition_query_is_required(self):
        tool = CourseSearchTool(MagicMock())
        schema = tool.get_tool_definition()["input_schema"]
        assert "query" in schema["required"]

    def test_get_tool_definition_optional_fields_present(self):
        tool = CourseSearchTool(MagicMock())
        props = tool.get_tool_definition()["input_schema"]["properties"]
        assert "course_name" in props
        assert "lesson_number" in props


# ===========================================================================
# ToolManager tests
# ===========================================================================

class TestToolManager:

    def test_register_and_get_definitions(self):
        tool = CourseSearchTool(_make_store())
        tm = ToolManager()
        tm.register_tool(tool)
        defs = tm.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "search_course_content"

    def test_execute_known_tool_returns_string(self):
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(_make_store()))
        result = tm.execute_tool("search_course_content", query="python")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_unknown_tool_returns_error_message(self):
        tm = ToolManager()
        result = tm.execute_tool("nonexistent_tool", query="test")
        assert "nonexistent_tool" in result
        assert "not found" in result.lower()

    def test_get_last_sources_returns_populated_list_after_execute(self):
        store = _make_store(lesson_link="https://example.com/lesson/1")
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(store))
        tm.execute_tool("search_course_content", query="python")
        sources = tm.get_last_sources()
        assert len(sources) >= 1

    def test_reset_sources_clears_last_sources(self):
        store = _make_store(lesson_link="https://example.com/lesson/1")
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(store))
        tm.execute_tool("search_course_content", query="python")
        assert len(tm.get_last_sources()) >= 1
        tm.reset_sources()
        assert tm.get_last_sources() == []

    def test_get_last_sources_returns_empty_when_no_tools_registered(self):
        tm = ToolManager()
        assert tm.get_last_sources() == []
