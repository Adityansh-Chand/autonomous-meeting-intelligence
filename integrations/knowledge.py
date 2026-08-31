"""Publish extracted meeting outcomes into the shared knowledge base.

A decision recorded in a meeting is exactly the kind of thing someone later
searches for and cannot find -- "what did we decide about the migration plan"
is not answerable from a policy corpus. Indexing meeting outcomes into the
retrieval service makes it answerable from the same endpoint that answers
everything else.

Optional and non-blocking, like every other integration here: if the retrieval
service is unset or down, analysis still returns normally and the response says
the publish did not happen.

    RAG_API_URL          retrieval service base URL
    INTEGRATION_API_KEY  X-API-Key, when the retrieval service requires one
    PUBLISH_TO_RAG       set to "0" to disable publishing while still configured
"""
import os

from integrations.client import OK, ServiceClient

# See the note in the operations service: call the versioned path so a provider
# can evolve behind /v2 without breaking this consumer on the same deploy.
API = "/v1"

_client = None


def rag_client():
    global _client
    if _client is None:
        _client = ServiceClient(
            "rag",
            base_url=os.getenv("RAG_API_URL", ""),
            api_key=os.getenv("INTEGRATION_API_KEY") or None,
        )
    return _client


def reset_client():
    """Test hook -- re-reads environment and clears breaker state."""
    global _client
    _client = None


def publishing_enabled():
    return os.getenv("PUBLISH_TO_RAG", "1") != "0" and rag_client().configured


def status():
    return {**rag_client().status(), "publishing_enabled": publishing_enabled()}


def render_document(meeting_id, summary):
    """Flatten a meeting result into one searchable passage.

    Owners and due dates are kept inline rather than dropped, because "who owns
    the migration plan" is the question people actually ask.
    """
    lines = [summary.summary]
    for decision in summary.decisions:
        lines.append(f"Decision: {decision}.")
    for item in summary.action_items:
        owner = item.owner or "unassigned"
        due = f", due {item.due_date}" if item.due_date else ""
        lines.append(f"Action item ({owner}{due}): {item.task}.")
    return " ".join(lines)


def publish(meeting_id, summary, request_id=None, title=None):
    """Index a meeting result. Returns a dict describing what happened."""
    if not publishing_enabled():
        return {"published": False, "outcome": "not_configured"}

    if not summary.decisions and not summary.action_items:
        # Nothing worth retrieving later; publishing chatter would pollute the corpus.
        return {"published": False, "outcome": "nothing_to_publish"}

    payload = {
        "doc_id": f"meeting:{meeting_id}",
        "title": title or f"Meeting {meeting_id}",
        "text": render_document(meeting_id, summary),
        "source": "autonomous-meeting-intelligence",
    }
    data, outcome = rag_client().post(f"{API}/documents", payload, request_id=request_id)
    return {
        "published": outcome == OK,
        "outcome": outcome,
        "doc_id": payload["doc_id"],
        "chunks_added": (data or {}).get("chunks_added"),
    }
