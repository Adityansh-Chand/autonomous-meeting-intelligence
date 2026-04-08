
# Autonomous Meeting Intelligence

LLM-powered structured transcript understanding pipeline.

## Architecture

```mermaid
flowchart LR
Transcript --> Chunker
Chunker --> Extractor
Extractor --> Structurer
Structurer --> Validator
``

## Pipeline
transcript → chunk → extract → structure → validate

### Highlights
schema validation
structured extraction
modular NLP pipeline

## License
MIT
