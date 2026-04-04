# Autonomous Meeting Intelligence

LLM-based meeting understanding and structured extraction system.

## Architecture

```mermaid
flowchart LR

Transcript --> Chunker
Chunker --> Embedding
Embedding --> Retrieval
Retrieval --> Summarizer
Summarizer --> ActionExtractor
ActionExtractor --> StructuredOutput
```


---

# License

MIT License
