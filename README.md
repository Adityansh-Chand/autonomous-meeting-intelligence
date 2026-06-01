# Autonomous Meeting Intelligence

Transcript understanding service that chunks meeting text, extracts decisions
and action items, and validates the output against a structured schema.

## Pipeline

```mermaid
flowchart LR
  Transcript --> Chunker
  Chunker --> Extractor
  Extractor --> Schema
  Schema --> Evaluation
```

## API

- `GET /health`
- `GET /metrics`
- `GET /events` protected when `API_KEY` is set
- `POST /analyze`

See `DEMO.md` for terminal demo steps, curl commands, and sample request/response files.

Example:

```json
{
  "transcript": "Decision: approve launch. Priya will follow up with legal by Friday."
}
```

Set `API_KEY` to require `X-API-Key` on analysis/event endpoints.
Set `APP_DB_PATH` to control the SQLite event database location.

## Run

```bash
pip install -r requirements.txt
python -m pytest -q
python evaluation/evaluate.py
uvicorn api.server:app --reload --port 8000
```

With the server running, use a second terminal for the smoke check:

```bash
python scripts/smoke_test.py
```

Docker:

```bash
cp .env.example .env
docker compose up --build
```

Kubernetes manifests live in `k8s/deployment.yaml` and include probes, resource
limits, a Service, and a PVC for the SQLite event store. The default manifest
uses one replica because SQLite is the default event store.

Dockerfile, Docker Compose, and Kubernetes configuration are validated by static
inspection/YAML parsing in this workspace. Runtime container and cluster
validation remains a CI or cloud-environment step.

## Highlights

- Speaker/transcript-friendly chunking.
- Decision extraction.
- Action item extraction with owner and due date.
- Pydantic schema validation.
- Structure evaluation over sample transcripts.
- SQLite event audit trail for analysis metadata.
- GitHub Actions CI for tests, eval, and container build.
- Production data contract in `datasets/production_schema.json`.

## License

MIT
