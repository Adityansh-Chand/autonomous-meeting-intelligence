import re


def chunk_transcript(text, max_chars=700):
    utterances = [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", text) if part.strip()]
    chunks = []
    current = ""

    for utterance in utterances:
        candidate = f"{current} {utterance}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = utterance
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks
