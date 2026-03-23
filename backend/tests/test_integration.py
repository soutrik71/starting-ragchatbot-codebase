"""
Integration tests for the full RAG system.

Scope:
  Group 1 — Search layer   : real ChromaDB (tmp_path) + real embeddings
  Group 2 — Tool execution : CourseSearchTool against real ChromaDB
  Group 3 — Full RAG query : RAGSystem with mocked Anthropic client
  Group 4 — Session & analytics

No real Anthropic API calls are made; the client is replaced with a MagicMock.
"""

import types
import pytest
from unittest.mock import MagicMock, patch

from tests.conftest import make_direct_response, make_tool_use_response

# ---------------------------------------------------------------------------
# Config factory (avoids importing the real config which reads .env)
# ---------------------------------------------------------------------------

def _make_config(chroma_path: str):
    cfg = types.SimpleNamespace(
        ANTHROPIC_API_KEY="sk-ant-test-key",
        ANTHROPIC_MODEL="claude-test-model",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        MAX_RESULTS=3,
        MAX_HISTORY=2,
        CHROMA_PATH=chroma_path,
    )
    return cfg


# ===========================================================================
# Group 1: Search layer (real ChromaDB, real embeddings)
# ===========================================================================

class TestSearchLayer:

    def test_add_course_and_search_returns_relevant_chunks(
        self, vector_store, sample_course_file
    ):
        from document_processor import DocumentProcessor
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
        course, chunks = processor.process_course_document(sample_course_file)
        vector_store.add_course_metadata(course)
        vector_store.add_course_content(chunks)

        results = vector_store.search("Python programming language")

        assert not results.is_empty()
        assert len(results.documents) > 0
        # Every result must belong to the indexed course
        for meta in results.metadata:
            assert meta["course_title"] == course.title

    def test_search_with_course_name_filter_returns_only_matching_course(
        self, vector_store, sample_course_file, sample_course_file_2
    ):
        from document_processor import DocumentProcessor
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        course1, chunks1 = processor.process_course_document(sample_course_file)
        course2, chunks2 = processor.process_course_document(sample_course_file_2)

        vector_store.add_course_metadata(course1)
        vector_store.add_course_content(chunks1)
        vector_store.add_course_metadata(course2)
        vector_store.add_course_content(chunks2)

        # Filter by the Python course only
        results = vector_store.search("programming", course_name="Python")

        assert not results.is_empty()
        for meta in results.metadata:
            assert "Python" in meta["course_title"]

    def test_course_links_retrievable_after_indexing(
        self, vector_store, sample_course_file
    ):
        from document_processor import DocumentProcessor
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
        course, chunks = processor.process_course_document(sample_course_file)
        vector_store.add_course_metadata(course)
        vector_store.add_course_content(chunks)

        course_link = vector_store.get_course_link(course.title)
        lesson_link = vector_store.get_lesson_link(course.title, 1)

        assert course_link == "https://www.example.com/courses/python-intro"
        assert lesson_link == "https://www.example.com/courses/python-intro/lesson/1"


# ===========================================================================
# Group 2: Tool execution (CourseSearchTool + ToolManager with real store)
# ===========================================================================

class TestToolExecution:

    def test_course_search_tool_returns_content_and_populates_sources(
        self, populated_vector_store
    ):
        from search_tools import CourseSearchTool
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute(query="Python programming language")

        assert isinstance(result, str)
        assert len(result) > 0
        assert len(tool.last_sources) >= 1

    def test_tool_manager_routes_to_search_tool(self, populated_vector_store):
        from search_tools import CourseSearchTool, ToolManager
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(populated_vector_store))

        result = tm.execute_tool("search_course_content", query="variables data types")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_sources_contain_markdown_links_after_search(self, populated_vector_store):
        """Sources should be formatted as [label](url) because the fixture has lesson links."""
        from search_tools import CourseSearchTool
        tool = CourseSearchTool(populated_vector_store)
        tool.execute(query="Python programming")

        import re
        for source in tool.last_sources:
            # Either a markdown link or plain text — no broken []() syntax
            if source.startswith("["):
                assert re.match(r"\[.+\]\(https?://.+\)", source), (
                    f"Malformed markdown link in source: {source}"
                )


# ===========================================================================
# Group 3: Full RAG query (RAGSystem with mocked Anthropic)
# ===========================================================================

class TestRAGQuery:

    @pytest.fixture
    def rag_system(self, tmp_path, sample_course_file, mocker):
        """RAGSystem with real ChromaDB + mocked Anthropic client."""
        mocker.patch("ai_generator.anthropic.Anthropic", return_value=MagicMock())
        from rag_system import RAGSystem
        cfg = _make_config(str(tmp_path / "chroma"))
        rag = RAGSystem(cfg)

        # Pre-index a real course
        rag.add_course_document(sample_course_file)
        return rag

    def test_query_with_tool_use_returns_answer_and_sources(self, rag_system, mocker):
        """AI triggers tool use → real search runs → final answer returned with sources."""
        tool_call = make_tool_use_response(
            "search_course_content", {"query": "Python programming"}, tool_id="t1"
        )
        final = make_direct_response("Python is a high-level language.")
        rag_system.ai_generator.client.messages.create.side_effect = [tool_call, final]

        session_id = rag_system.session_manager.create_session()
        answer, sources = rag_system.query("What is Python?", session_id)

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(sources, list)
        assert len(sources) >= 1  # tool was called → sources populated

    def test_query_direct_answer_returns_empty_sources(self, rag_system):
        """AI answers directly without calling any tool → sources should be empty."""
        rag_system.ai_generator.client.messages.create.return_value = make_direct_response(
            "I can answer that without searching."
        )

        session_id = rag_system.session_manager.create_session()
        answer, sources = rag_system.query("What is 2 + 2?", session_id)

        assert answer == "I can answer that without searching."
        assert sources == []

    def test_query_stores_exchange_in_session_history(self, rag_system):
        """After a query, the session history must contain the exchange."""
        rag_system.ai_generator.client.messages.create.return_value = make_direct_response(
            "Python was created by Guido van Rossum."
        )

        session_id = rag_system.session_manager.create_session()
        rag_system.query("Who created Python?", session_id)

        history = rag_system.session_manager.get_conversation_history(session_id)
        assert "Who created Python?" in history
        assert "Guido van Rossum" in history


# ===========================================================================
# Group 4: Session management and analytics
# ===========================================================================

class TestSessionManagement:

    def test_two_sessions_are_isolated(self):
        from session_manager import SessionManager
        sm = SessionManager(max_history=5)
        s1 = sm.create_session()
        s2 = sm.create_session()

        sm.add_exchange(s1, "hello from s1", "response for s1")

        # s2 must have no history
        history_s2 = sm.get_conversation_history(s2)
        assert history_s2 is None or "s1" not in (history_s2 or "")

    def test_session_history_respects_max_history(self):
        from session_manager import SessionManager
        sm = SessionManager(max_history=1)  # keep only 1 turn = 2 messages
        sid = sm.create_session()

        sm.add_exchange(sid, "first question", "first answer")
        sm.add_exchange(sid, "second question", "second answer")

        # With max_history=1 only the most recent turn is kept
        history = sm.get_conversation_history(sid)
        assert "second question" in history
        # Oldest turn trimmed
        assert "first question" not in history

    def test_clear_session_wipes_history(self):
        from session_manager import SessionManager
        sm = SessionManager(max_history=5)
        sid = sm.create_session()
        sm.add_exchange(sid, "question", "answer")

        sm.clear_session(sid)
        history = sm.get_conversation_history(sid)
        assert history is None or history == ""


class TestCourseAnalytics:

    def test_analytics_reflects_indexed_courses(
        self, tmp_path, sample_docs_folder, mocker
    ):
        mocker.patch("ai_generator.anthropic.Anthropic", return_value=MagicMock())
        from rag_system import RAGSystem
        cfg = _make_config(str(tmp_path / "chroma"))
        rag = RAGSystem(cfg)

        courses_added, _ = rag.add_course_folder(sample_docs_folder, clear_existing=True)

        analytics = rag.get_course_analytics()
        assert analytics["total_courses"] == courses_added
        assert len(analytics["course_titles"]) == courses_added

    def test_analytics_total_courses_is_zero_on_empty_store(
        self, tmp_path, mocker
    ):
        mocker.patch("ai_generator.anthropic.Anthropic", return_value=MagicMock())
        from rag_system import RAGSystem
        cfg = _make_config(str(tmp_path / "chroma"))
        rag = RAGSystem(cfg)

        analytics = rag.get_course_analytics()
        assert analytics["total_courses"] == 0
        assert analytics["course_titles"] == []
