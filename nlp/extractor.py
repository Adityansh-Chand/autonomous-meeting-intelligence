import re

from schema.output_schema import ActionItem, MeetingSummary


DATE_PATTERN = r"(?:by|before|on)\s+([A-Z][a-z]+day|tomorrow|next week|EOD|end of week|\d{4}-\d{2}-\d{2})"


def _sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def extract_actions(text):
    return [item.task for item in extract_action_items(text)]


def extract_action_items(text):
    items = []
    for sentence in _sentences(text):
        lower = sentence.lower()
        if not any(term in lower for term in ["action:", "follow up", "will", "todo"]):
            continue

        owner = None
        owner_match = re.match(r"([A-Z][a-z]+)\s+(?:will|to)\s+", sentence)
        if owner_match:
            owner = owner_match.group(1)

        due_date = None
        date_match = re.search(DATE_PATTERN, sentence)
        if date_match:
            due_date = date_match.group(1)

        task = re.sub(r"^Action:\s*", "", sentence, flags=re.IGNORECASE)
        items.append(ActionItem(task=task, owner=owner, due_date=due_date, source_text=sentence))

    return items


def extract_decisions(text):
    decisions = []
    for sentence in _sentences(text):
        lower = sentence.lower()
        if "decision:" in lower:
            decisions.append(re.sub(r"^Decision:\s*", "", sentence, flags=re.IGNORECASE))
        elif "decided to" in lower or "approved" in lower:
            decisions.append(sentence)
    return decisions


def summarize(text):
    sentences = _sentences(text)
    if not sentences:
        return "No transcript content was provided."

    first = sentences[0]
    return first if len(first) <= 180 else f"{first[:177]}..."


def extract_meeting_intelligence(text):
    return MeetingSummary(
        summary=summarize(text),
        decisions=extract_decisions(text),
        action_items=extract_action_items(text),
    )
