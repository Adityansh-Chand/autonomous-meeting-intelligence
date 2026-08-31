
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from monitoring.drift import DriftMonitor
from monitoring.metrics import metrics
from integrations.knowledge import publish, status as knowledge_status
from nlp.extractor import classify_transcript_sentences, extract_meeting_intelligence, load_classifier
from schema.output_schema import to_dict
from utils.security import current_request_id, request_id_middleware, require_api_key
from utils.storage import recent_events, save_event

app = FastAPI(title="Autonomous Meeting Intelligence", version="1.0.0")
app.middleware("http")(request_id_middleware)

API_VERSION = "v1"

# Data endpoints live on a router so they can be served at BOTH /v1/... and the
# original unversioned paths from a single definition. Without a version prefix
# there is no way to change a response shape without breaking every consumer on
# the same deploy -- the contract checks in the portfolio repo detect that
# breakage, they do not prevent it.
#
# The unversioned alias is kept because consumers already call it. It is the
# deprecation path, not a permanent second interface.
api = APIRouter()

# The mix of sentence classes this extractor is producing, against the mix seen at
# training time. If transcripts stop containing decisions, that shows up here
# before it shows up as an empty summary somebody complains about.
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "models" / "artifacts"
drift_monitor = DriftMonitor.from_file(
    ARTIFACT_DIR / "drift_reference.json", name="sentence_class_mix"
)


class TranscriptRequest(BaseModel):
    transcript: str = Field(..., min_length=1, max_length=100000)
    meeting_id: str = Field("adhoc", max_length=200)
    title: str = Field("", max_length=500)




@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    metrics.increment("http_errors_total")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "path": str(request.url.path)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    metrics.increment("validation_errors_total")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request",
            "details": exc.errors(),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    metrics.increment("unhandled_errors_total")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "path": str(request.url.path)},
    )


@app.get("/")
def health():
    return {"status": "running"}


@app.get("/health")
def health_check():
    """Health plus the identity of the classifier actually loaded."""
    _, metadata = load_classifier()
    return {
        "status": "running",
        "model": {
            "model_type": metadata["model_type"],
            "data_source": metadata["data_source"],
            "split": metadata["split"],
            "test_macro_f1": metadata["test_macro_f1"],
            "keyword_baseline_macro_f1": metadata["legacy_keyword_baseline"]["macro_f1"],
        },
        "knowledge_base": knowledge_status(),
    }


@app.get("/metrics")
def metrics_endpoint():
    return metrics.snapshot()


@api.get("/events", dependencies=[Depends(require_api_key)])
def events(limit: int = 20, request_id: str | None = None):
    """Recent events, optionally narrowed to one request id.

    `request_id` is what makes this endpoint a trace source rather than a log
    tail: the portfolio's scripts/trace.py asks all five services the same
    question and joins the answers into one timeline.
    """
    return {"events": recent_events(limit=min(limit, 100), request_id=request_id)}


@api.post("/analyze", dependencies=[Depends(require_api_key)])
def analyze(request: TranscriptRequest, http_request: Request):
    metrics.increment("analyses_total")
    summary = extract_meeting_intelligence(request.transcript)
    result = to_dict(summary)

    # Publish outcomes to the shared knowledge base so they become searchable.
    # Optional: if retrieval is unset or down, analysis still returns normally
    # and the response says the publish did not happen.
    request_id = current_request_id(http_request)
    result["knowledge_publish"] = publish(
        request.meeting_id, summary, request_id=request_id, title=request.title or None
    )
    if request_id:
        result["request_id"] = request_id
    # Every sentence's label, not just the extracted ones: the reference is the
    # full class mix, and `neither` is 58% of it.
    for label in classify_transcript_sentences(request.transcript):
        drift_monitor.observe(label)

    save_event(
        "meeting_analysis",
        {
            "summary": result["summary"],
            "decision_count": len(result["decisions"]),
            "action_count": len(result["action_items"]),
        },
        request_id,
    )
    return result


@api.get("/drift", dependencies=[Depends(require_api_key)])
def drift():
    """Are transcripts still the shape this classifier was fitted on?"""
    return drift_monitor.report()


@app.get("/version")
def version():
    """What this service speaks, so a consumer can check rather than assume."""
    return {
        "service": "autonomous-meeting-intelligence",
        "current": API_VERSION,
        "supported": [API_VERSION],
        "unversioned_alias": {
            "status": "deprecated",
            "note": ("the same endpoints are served without a /v1 prefix for "
                     "consumers that predate versioning; new callers should use "
                     f"/{API_VERSION}"),
        },
    }


# Mounted twice, one set of handlers. The alias is hidden from the schema so the
# generated docs show one interface rather than two identical ones.
app.include_router(api, prefix=f"/{API_VERSION}")
app.include_router(api, include_in_schema=False)
