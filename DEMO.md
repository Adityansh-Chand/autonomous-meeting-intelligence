# Demo

This demo shows the meeting intelligence service extracting a summary,
decisions, action items, metrics, and an audit event from a transcript.

## Run Locally

Terminal 1:

```bash
pip install -r requirements.txt
uvicorn api.server:app --reload --port 8000
```

Terminal 2:

```bash
python scripts/smoke_test.py
```

To demo protected endpoints, start with an API key:

```bash
API_KEY=demo-key uvicorn api.server:app --reload --port 8000
```

## Curl Walkthrough

Root:

```bash
curl http://localhost:8000/
```

Health:

```bash
curl http://localhost:8000/health
```

Metrics:

```bash
curl http://localhost:8000/metrics
```

Analyze transcript:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d @examples/requests/analyze.json
```

Events when `API_KEY` is set:

```bash
curl http://localhost:8000/events \
  -H "X-API-Key: demo-key"
```

Protected analysis when `API_KEY` is set:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-key" \
  -d @examples/requests/analyze.json
```

## Sample Files

- Request: `examples/requests/analyze.json`
- Responses: `examples/responses/root.json`, `health.json`, `metrics.json`, `analyze.json`, `events.json`
