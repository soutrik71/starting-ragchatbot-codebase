"""
API-level tests for the FastAPI HTTP endpoints.

Endpoints under test:
  POST /api/query       → QueryRequest / QueryResponse
  GET  /api/courses     → CourseStats
  POST /api/new-session → NewSessionRequest / NewSessionResponse

Strategy:
  - Patch `rag_system.RAGSystem` at module load time so app.py's module-level
    `rag_system = RAGSystem(config)` receives a MagicMock — prevents ChromaDB
    client creation and embedding-model loading on import.
  - Replace `app.rag_system` per-test to control return values and assert
    call-site behaviour precisely.
  - Use Starlette TestClient (synchronous) for HTTP calls.
  - No real Anthropic API calls or ChromaDB I/O in any test.
"""

import sys
import uuid
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Module-level bootstrap
#
# Patch RAGSystem BEFORE importing app so the module-level constructor call
# `rag_system = RAGSystem(config)` in app.py returns a MagicMock instead of
# triggering real ChromaDB / embedding-model initialisation.
# ---------------------------------------------------------------------------
with patch("rag_system.RAGSystem") as _patched_rag_cls:
    _patched_rag_cls.return_value = MagicMock()
    sys.modules.pop("app", None)   # force a clean import under the patch
    from app import app            # noqa: E402

from starlette.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------------------
# Expected field sets (single source of truth for schema assertions)
# ---------------------------------------------------------------------------

QUERY_RESPONSE_FIELDS    = {"answer", "sources", "session_id"}
COURSES_RESPONSE_FIELDS  = {"total_courses", "course_titles"}
SESSION_RESPONSE_FIELDS  = {"session_id", "conversation_id"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag(mocker):
    """
    Replace `app.rag_system` with a fully-configured MagicMock.

    Default happy-path stubs:
      - query()               → ("Here is your answer.", ["[Lesson 1](https://example.com/lesson/1)"])
      - get_course_analytics() → {total_courses: 2, course_titles: [...]}
      - session_manager.create_session() → "sess-generated-123"
    """
    mock = MagicMock()
    mock.query.return_value = (
        "Here is your answer.",
        ["[Lesson 1](https://example.com/lesson/1)"],
    )
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Intro to Python", "Advanced ML"],
    }
    mock.session_manager.create_session.return_value = "sess-generated-123"
    mock.session_manager.clear_session.return_value = None
    mocker.patch("app.rag_system", mock)
    return mock


@pytest.fixture
def client(mock_rag):
    """TestClient wired to the FastAPI app with a mocked RAG system."""
    return TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# POST /api/query — request-schema validation
# ===========================================================================

class TestQueryRequestSchema:

    def test_missing_query_field_returns_422(self, client):
        resp = client.post("/api/query", json={})
        assert resp.status_code == 422

    def test_missing_query_field_error_names_query(self, client):
        resp = client.post("/api/query", json={})
        error_fields = [e["loc"][-1] for e in resp.json()["detail"]]
        assert "query" in error_fields

    def test_empty_json_body_returns_422(self, client):
        resp = client.post(
            "/api/query",
            content=b"{}",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_malformed_json_returns_422(self, client):
        resp = client.post(
            "/api/query",
            content=b"{bad json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_session_id_is_optional(self, client):
        resp = client.post("/api/query", json={"query": "What is Python?"})
        assert resp.status_code == 200

    def test_session_id_accepted_when_provided(self, client):
        resp = client.post("/api/query", json={"query": "What is Python?", "session_id": "abc"})
        assert resp.status_code == 200

    def test_extra_fields_are_silently_ignored(self, client):
        resp = client.post("/api/query", json={"query": "Hi", "unexpected": "value"})
        assert resp.status_code == 200


# ===========================================================================
# POST /api/query — response-schema validation
# ===========================================================================

class TestQueryResponseSchema:

    def test_response_contains_all_required_fields(self, client):
        body = client.post("/api/query", json={"query": "What is Python?"}).json()
        assert QUERY_RESPONSE_FIELDS.issubset(body.keys())

    def test_answer_is_non_empty_string(self, client):
        body = client.post("/api/query", json={"query": "What is Python?"}).json()
        assert isinstance(body["answer"], str)
        assert len(body["answer"]) > 0

    def test_sources_is_list(self, client):
        body = client.post("/api/query", json={"query": "What is Python?"}).json()
        assert isinstance(body["sources"], list)

    def test_sources_elements_are_strings(self, client, mock_rag):
        mock_rag.query.return_value = ("Answer", ["src1", "src2"])
        body = client.post("/api/query", json={"query": "What is Python?"}).json()
        for item in body["sources"]:
            assert isinstance(item, str)

    def test_sources_can_be_empty_list(self, client, mock_rag):
        mock_rag.query.return_value = ("Direct answer.", [])
        body = client.post("/api/query", json={"query": "2 + 2?"}).json()
        assert body["sources"] == []

    def test_session_id_is_string(self, client):
        body = client.post("/api/query", json={"query": "Hi"}).json()
        assert isinstance(body["session_id"], str)

    def test_answer_matches_rag_system_output(self, client, mock_rag):
        mock_rag.query.return_value = ("Python is a high-level language.", [])
        body = client.post("/api/query", json={"query": "What is Python?"}).json()
        assert body["answer"] == "Python is a high-level language."

    def test_sources_match_rag_system_output(self, client, mock_rag):
        links = ["[Lesson 1](https://example.com/1)", "[Lesson 2](https://example.com/2)"]
        mock_rag.query.return_value = ("Answer", links)
        body = client.post("/api/query", json={"query": "Tell me about loops"}).json()
        assert body["sources"] == links


# ===========================================================================
# POST /api/query — routing and side-effect behaviour
# ===========================================================================

class TestQueryBehavior:

    def test_provided_session_id_is_forwarded_to_rag(self, client, mock_rag):
        mock_rag.query.return_value = ("Answer", [])
        client.post("/api/query", json={"query": "Hi", "session_id": "existing-sess"})
        mock_rag.query.assert_called_once_with("Hi", "existing-sess")

    def test_new_session_created_when_no_session_id(self, client, mock_rag):
        mock_rag.query.return_value = ("Answer", [])
        client.post("/api/query", json={"query": "Hi"})
        mock_rag.session_manager.create_session.assert_called_once()

    def test_session_id_not_created_when_already_provided(self, client, mock_rag):
        mock_rag.query.return_value = ("Answer", [])
        client.post("/api/query", json={"query": "Hi", "session_id": "existing"})
        mock_rag.session_manager.create_session.assert_not_called()

    def test_response_session_id_echoes_provided_session(self, client, mock_rag):
        mock_rag.query.return_value = ("Answer", [])
        body = client.post("/api/query", json={"query": "Hi", "session_id": "my-sess"}).json()
        assert body["session_id"] == "my-sess"

    def test_response_session_id_is_generated_when_none_given(self, client, mock_rag):
        mock_rag.session_manager.create_session.return_value = "fresh-id-999"
        mock_rag.query.return_value = ("Answer", [])
        body = client.post("/api/query", json={"query": "Hi"}).json()
        assert body["session_id"] == "fresh-id-999"


# ===========================================================================
# POST /api/query — error handling
# ===========================================================================

class TestQueryErrorHandling:

    def test_rag_exception_returns_500(self, client, mock_rag):
        mock_rag.query.side_effect = RuntimeError("Vector DB failure")
        resp = client.post("/api/query", json={"query": "What is Python?"})
        assert resp.status_code == 500

    def test_500_response_contains_detail_field(self, client, mock_rag):
        mock_rag.query.side_effect = RuntimeError("Vector DB failure")
        resp = client.post("/api/query", json={"query": "What is Python?"})
        assert "detail" in resp.json()

    def test_500_detail_includes_error_message(self, client, mock_rag):
        mock_rag.query.side_effect = RuntimeError("Vector DB failure")
        resp = client.post("/api/query", json={"query": "What is Python?"})
        assert "Vector DB failure" in resp.json()["detail"]

    def test_session_creation_exception_propagates_as_500(self, client, mock_rag):
        mock_rag.session_manager.create_session.side_effect = RuntimeError("session store down")
        resp = client.post("/api/query", json={"query": "Hi"})
        assert resp.status_code == 500


# ===========================================================================
# GET /api/courses — response-schema validation
# ===========================================================================

class TestCoursesResponseSchema:

    def test_returns_200(self, client):
        assert client.get("/api/courses").status_code == 200

    def test_response_contains_all_required_fields(self, client):
        body = client.get("/api/courses").json()
        assert COURSES_RESPONSE_FIELDS.issubset(body.keys())

    def test_total_courses_is_int(self, client):
        assert isinstance(client.get("/api/courses").json()["total_courses"], int)

    def test_course_titles_is_list(self, client):
        assert isinstance(client.get("/api/courses").json()["course_titles"], list)

    def test_course_titles_elements_are_strings(self, client):
        for title in client.get("/api/courses").json()["course_titles"]:
            assert isinstance(title, str)

    def test_total_courses_equals_titles_length(self, client):
        body = client.get("/api/courses").json()
        assert body["total_courses"] == len(body["course_titles"])


# ===========================================================================
# GET /api/courses — data accuracy
# ===========================================================================

class TestCoursesData:

    def test_reflects_current_indexed_courses(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Course A", "Course B", "Course C"],
        }
        body = client.get("/api/courses").json()
        assert body["total_courses"] == 3
        assert set(body["course_titles"]) == {"Course A", "Course B", "Course C"}

    def test_empty_store_returns_zero_total(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {"total_courses": 0, "course_titles": []}
        body = client.get("/api/courses").json()
        assert body["total_courses"] == 0
        assert body["course_titles"] == []

    def test_single_course_returns_count_of_one(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Intro to Python"],
        }
        body = client.get("/api/courses").json()
        assert body["total_courses"] == 1


# ===========================================================================
# GET /api/courses — error handling
# ===========================================================================

class TestCoursesErrorHandling:

    def test_analytics_exception_returns_500(self, client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("analytics DB error")
        assert client.get("/api/courses").status_code == 500

    def test_500_contains_detail_field(self, client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("analytics DB error")
        assert "detail" in client.get("/api/courses").json()

    def test_500_detail_includes_error_message(self, client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("analytics DB error")
        resp = client.get("/api/courses")
        assert "analytics DB error" in resp.json()["detail"]


# ===========================================================================
# POST /api/new-session — request-schema validation
# ===========================================================================

class TestNewSessionRequestSchema:

    def test_empty_body_returns_200(self, client):
        assert client.post("/api/new-session", json={}).status_code == 200

    def test_session_id_is_optional(self, client):
        assert client.post("/api/new-session", json={}).status_code == 200

    def test_session_id_accepted_when_provided(self, client):
        assert client.post("/api/new-session", json={"session_id": "old-session"}).status_code == 200

    def test_extra_fields_are_ignored(self, client):
        assert client.post("/api/new-session", json={"unknown": "value"}).status_code == 200


# ===========================================================================
# POST /api/new-session — response-schema validation
# ===========================================================================

class TestNewSessionResponseSchema:

    def test_response_contains_all_required_fields(self, client):
        body = client.post("/api/new-session", json={}).json()
        assert SESSION_RESPONSE_FIELDS.issubset(body.keys())

    def test_session_id_is_string(self, client):
        body = client.post("/api/new-session", json={}).json()
        assert isinstance(body["session_id"], str)

    def test_conversation_id_is_string(self, client):
        body = client.post("/api/new-session", json={}).json()
        assert isinstance(body["conversation_id"], str)

    def test_conversation_id_is_valid_uuid(self, client):
        body = client.post("/api/new-session", json={}).json()
        uuid.UUID(body["conversation_id"])  # raises ValueError if invalid

    def test_conversation_id_is_uuid4_version(self, client):
        body = client.post("/api/new-session", json={}).json()
        assert uuid.UUID(body["conversation_id"]).version == 4


# ===========================================================================
# POST /api/new-session — routing and side-effect behaviour
# ===========================================================================

class TestNewSessionBehavior:

    def test_clears_old_session_when_provided(self, client, mock_rag):
        client.post("/api/new-session", json={"session_id": "old-sess"})
        mock_rag.session_manager.clear_session.assert_called_once_with("old-sess")

    def test_does_not_clear_when_no_session_id_given(self, client, mock_rag):
        client.post("/api/new-session", json={})
        mock_rag.session_manager.clear_session.assert_not_called()

    def test_always_creates_a_new_session(self, client, mock_rag):
        client.post("/api/new-session", json={})
        mock_rag.session_manager.create_session.assert_called_once()

    def test_returned_session_id_comes_from_session_manager(self, client, mock_rag):
        mock_rag.session_manager.create_session.return_value = "brand-new-sess"
        body = client.post("/api/new-session", json={}).json()
        assert body["session_id"] == "brand-new-sess"

    def test_successive_calls_produce_different_conversation_ids(self, client):
        cid1 = client.post("/api/new-session", json={}).json()["conversation_id"]
        cid2 = client.post("/api/new-session", json={}).json()["conversation_id"]
        assert cid1 != cid2

    def test_clear_then_create_order_is_respected(self, client, mock_rag):
        """clear_session must be called before create_session."""
        call_order = []
        mock_rag.session_manager.clear_session.side_effect = lambda s: call_order.append("clear")
        mock_rag.session_manager.create_session.side_effect = lambda: call_order.append("create") or "new"

        client.post("/api/new-session", json={"session_id": "old"})

        assert call_order == ["clear", "create"]


# ===========================================================================
# POST /api/new-session — error handling
# ===========================================================================

class TestNewSessionErrorHandling:

    def test_session_manager_exception_returns_500(self, client, mock_rag):
        mock_rag.session_manager.create_session.side_effect = RuntimeError("session store down")
        assert client.post("/api/new-session", json={}).status_code == 500

    def test_500_contains_detail_field(self, client, mock_rag):
        mock_rag.session_manager.create_session.side_effect = RuntimeError("session store down")
        assert "detail" in client.post("/api/new-session", json={}).json()

    def test_500_detail_includes_error_message(self, client, mock_rag):
        mock_rag.session_manager.create_session.side_effect = RuntimeError("session store down")
        resp = client.post("/api/new-session", json={})
        assert "session store down" in resp.json()["detail"]

    def test_clear_session_exception_returns_500(self, client, mock_rag):
        mock_rag.session_manager.clear_session.side_effect = RuntimeError("clear failed")
        resp = client.post("/api/new-session", json={"session_id": "old"})
        assert resp.status_code == 500
