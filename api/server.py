
from fastapi import FastAPI
from pydantic import BaseModel, Field

from nlp.extractor import extract_meeting_intelligence
from schema.output_schema import to_dict

app = FastAPI()


class TranscriptRequest(BaseModel):
    transcript: str = Field(..., min_length=1)


@app.get("/")
def health():
    return {"status": "running"}


@app.get("/health")
def health_check():
    return {"status": "running"}


@app.post("/analyze")
def analyze(request: TranscriptRequest):
    return to_dict(extract_meeting_intelligence(request.transcript))
