
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from monitoring.metrics import metrics
from integrations.knowledge import publish, status as knowledge_status
from nlp.extractor import extract_meeting_intelligence, load_classifier
from schema.output_schema import to_dict
from utils.security import current_request_id, request_id_middleware, require_api_key
from utils.storage import recent_events, save_event

app = FastAPI(title="Autonomous Meeting Intelligence", version="1.0.0")
app.middleware("http")(request_id_middleware)


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


@app.get("/events", dependencies=[Depends(require_api_key)])
def events(limit: int = 20):
    return {"events": recent_events(limit=min(limit, 100))}


@app.post("/analyze", dependencies=[Depends(require_api_key)])
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
    save_event(
        "meeting_analysis",
        {
            "summary": result["summary"],
            "decision_count": len(result["decisions"]),
            "action_count": len(result["action_items"]),
        },
    )
    return result
