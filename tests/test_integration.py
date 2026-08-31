"""Publishing meeting outcomes to the shared knowledge base.

Optional, like every integration here: if retrieval is unset or down, analysis
still returns normally and the response says the publish did not happen.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from integrations.knowledge import publish, publishing_enabled, render_document, reset_client
from nlp.extractor import extract_meeting_intelligence

TRANSCRIPT = (
    "Ravi: morning everyone.\n"
    "Maya: we agreed to move forward with the migration plan.\n"
    "Priya will send the notes on the rollout by Friday."
)
EMPTY = "Ravi: morning everyone.\nMaya: sorry, could you repeat that."


class _Stub(BaseHTTPRequestHandler):
    received = []
    status_code = 200

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).received.append({"path": self.path, "body": body,
                                    "headers": dict(self.headers)})
        if type(self).status_code != 200:
            self.send_error(type(self).status_code)
            return
        payload = json.dumps({"doc_id": body.get("doc_id"), "chunks_added": 1}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub():
    _Stub.received = []
    _Stub.status_code = 200
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_publishing_is_off_by_default(monkeypatch):
    monkeypatch.delenv("RAG_API_URL", raising=False)
    reset_client()
    assert publishing_enabled() is False

    summary = extract_meeting_intelligence(TRANSCRIPT)
    assert publish("mtg_1", summary) == {"published": False, "outcome": "not_configured"}


def test_analysis_is_unaffected_when_publishing_is_off(monkeypatch):
    monkeypatch.delenv("RAG_API_URL", raising=False)
    reset_client()
    summary = extract_meeting_intelligence(TRANSCRIPT)
    assert summary.decisions
    assert summary.action_items


def test_publish_sends_a_searchable_document(stub, monkeypatch):
    monkeypatch.setenv("RAG_API_URL", stub)
    reset_client()

    summary = extract_meeting_intelligence(TRANSCRIPT)
    result = publish("mtg_0007", summary, request_id="req-xyz", title="Weekly sync")

    assert result["published"] is True
    assert result["doc_id"] == "meeting:mtg_0007"

    sent = _Stub.received[0]["body"]
    assert sent["source"] == "autonomous-meeting-intelligence"
    # Owner and due date must survive into the indexed text -- "who owns the
    # migration plan" is the question people actually ask later.
    assert "migration plan" in sent["text"]
    assert "Priya" in sent["text"]
    assert "Friday" in sent["text"]


def test_publish_propagates_the_request_id(stub, monkeypatch):
    monkeypatch.setenv("RAG_API_URL", stub)
    reset_client()
    publish("mtg_1", extract_meeting_intelligence(TRANSCRIPT), request_id="req-abc")

    forwarded = [
        value
        for entry in _Stub.received
        for key, value in entry["headers"].items()
        if key.lower() == "x-request-id"
    ]
    assert "req-abc" in forwarded


def test_meetings_with_nothing_recorded_are_not_published(stub, monkeypatch):
    """Indexing chatter would pollute the corpus for everyone else."""
    monkeypatch.setenv("RAG_API_URL", stub)
    reset_client()

    result = publish("mtg_2", extract_meeting_intelligence(EMPTY))
    assert result == {"published": False, "outcome": "nothing_to_publish"}
    assert _Stub.received == []


def test_a_failing_knowledge_base_does_not_break_analysis(stub, monkeypatch):
    monkeypatch.setenv("RAG_API_URL", stub)
    _Stub.status_code = 500
    reset_client()

    summary = extract_meeting_intelligence(TRANSCRIPT)
    result = publish("mtg_3", summary)

    assert result["published"] is False
    assert result["outcome"] in {"error", "timeout"}
    # The analysis itself is untouched.
    assert summary.decisions


def test_unreachable_knowledge_base_does_not_hang(monkeypatch):
    monkeypatch.setenv("RAG_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("INTEGRATION_TIMEOUT_SECONDS", "0.2")
    reset_client()

    started = time.monotonic()
    result = publish("mtg_4", extract_meeting_intelligence(TRANSCRIPT))
    assert result["published"] is False
    assert time.monotonic() - started < 2.0


def test_rendered_document_is_self_contained():
    summary = extract_meeting_intelligence(TRANSCRIPT)
    text = render_document("mtg_9", summary)
    assert "Decision:" in text
    assert "Action item" in text


def test_publish_uses_the_versioned_endpoint(stub, monkeypatch):
    """The consumer must ask for /v1, not the deprecated bare path.

    The stub accepts any POST path, so calling the wrong one would pass silently
    until the alias is removed. This asserts the version guarantee is used, not
    merely available.
    """
    monkeypatch.setenv("RAG_API_URL", stub)
    reset_client()

    publish("mtg_version_1", extract_meeting_intelligence(TRANSCRIPT))

    assert _Stub.received, "the publish never reached the retrieval service"
    paths = [entry["path"] for entry in _Stub.received]
    assert all(path.startswith("/v1/") for path in paths), (
        f"consumer used unversioned paths: {paths}"
    )
