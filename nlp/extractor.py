"""Extract decisions and action items from a transcript.

Sentence classification is **learned**. It replaces a keyword gate that required
one of "action:", "follow up", "will" or "todo" to appear, and treated any
sentence containing "decision:" or "decided to" as a decision. In the generated
corpus only ~4% of real decisions carry an explicit marker, so that gate misses
almost all of them -- and it fires on "I will grab a coffee".

Owner and due-date extraction stays **rule-based**. Pulling a name and a date out
of an already-classified sentence is slot filling, and patterns are a legitimate
tool for it. The difference from before is that the slots are now *measured*
against gold annotations rather than assumed correct -- see
`evaluation/evaluate.py`.

An optional LLM path is available through the provider-agnostic seam. It is off
by default and no reported metric uses it.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

from llm import client as llm
from schema.output_schema import ActionItem, MeetingSummary

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "models" / "artifacts"
CLASSIFIER_PATH = ARTIFACT_DIR / "sentence_classifier.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

_MISSING = (
    f"Sentence classifier not found at {CLASSIFIER_PATH}.\n"
    "Train it first:\n"
    "    python training/generate_transcripts.py\n"
    "    python training/train.py"
)

# Date expressions, longest-first so "next Tuesday" wins over "Tuesday".
DATE_PATTERN = re.compile(
    r"\b("
    r"\d{4}-\d{2}-\d{2}"
    r"|end of (?:the )?(?:week|month|quarter|day)"
    r"|next (?:week|month|quarter|Monday|Tuesday|Wednesday|Thursday|Friday)"
    r"|in (?:a|two|three|four) weeks?"
    r"|the \d{1,2}(?:st|nd|rd|th)"
    r"|tomorrow|today|EOD|EOW"
    r"|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
    r")\b",
    re.IGNORECASE,
)

# --- owner extraction -------------------------------------------------------
#
# The first version anchored on role cues and captured a single capitalised word.
# It recalled 0.3484, and the misses fell into four groups, all visible in the
# corpus rather than guessed at:
#
#   the platform team          no pattern matched a team at all
#   Dr. Marcus                 the title was dropped, yielding "Marcus"
#   Ravi Raman                 the surname was taken alone, or missed entirely
#   Chen has the action on     the cue phrase was not covered
#
# So an owner is now one of two shapes, and the cue phrases are enumerated around
# them rather than a single capitalised word being grabbed near a verb.

# A person: an optional title, a given name, and an optional surname.
PERSON = r"(?:Dr\.\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"

# A collective owner. Real action items are assigned to teams as often as to
# people, and "the platform team" is a complete answer to "who owns this".
TEAM = r"(?:the\s+[a-z]+\s+(?:team|group)|someone\s+from\s+[a-z]+)"

OWNER = f"(?:{TEAM}|{PERSON})"

# Ordered: the first match wins, so more specific cues come first. Each anchors on
# a phrase that assigns work, never on capitalisation alone -- "Marcus was on mute"
# names a person who owns nothing.
OWNER_PATTERNS = [
    # Owner first, followed by a phrase that commits them to something.
    re.compile(
        rf"({OWNER})\s+(?:will|is going to|agreed to|takes|owns"
        rf"|has the action|please|to\s+)"
    ),
    re.compile(rf"Action:\s*({OWNER})"),
    # Owner last, after a phrase that hands the work over.
    re.compile(
        rf"(?:belongs to|assigned to|assigning\s+.*?\s+to|goes to|over to"
        rf"|hand(?:ed)? to)\s+({OWNER})"
    ),
    re.compile(rf"that one's on\s+({OWNER})"),
    re.compile(
        rf"can\s+({OWNER})\s+(?:pick|take|confirm|draft|send|review|chase|do)"
    ),
    re.compile(rf"put\s+.*?on\s+({OWNER})"),
    re.compile(rf"from\s+({OWNER})"),
    re.compile(rf"on\s+({OWNER})\s+to"),
    re.compile(rf"with\s+({OWNER})\s+to"),
    re.compile(rf"need\s+({OWNER})\s+to"),
]

SPEAKER_PREFIX = re.compile(r"^\s*([A-Z][a-z]+)\s*:\s*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

LLM_PROMPT = (
    "Classify each meeting sentence as exactly one of: decision, action_item, "
    "neither. Reply with the label only."
)


@lru_cache(maxsize=1)
def load_classifier():
    if not CLASSIFIER_PATH.exists() or not METRICS_PATH.exists():
        raise FileNotFoundError(_MISSING)
    import joblib

    metadata = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return joblib.load(CLASSIFIER_PATH), metadata


def split_sentences(text):
    """Split a transcript into sentences, stripping speaker prefixes."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    return [SPEAKER_PREFIX.sub("", part).strip().rstrip(".") for part in parts if part]


def extract_owner(sentence):
    """First cue-anchored owner -- person or team -- or None.

    Returns None rather than guessing. An action item with no stated owner is a
    real and common thing, and inventing one is worse than reporting none.
    """
    for pattern in OWNER_PATTERNS:
        match = pattern.search(sentence)
        if match:
            return match.group(1).strip()
    return None


def extract_due_date(sentence):
    match = DATE_PATTERN.search(sentence)
    return match.group(1) if match else None


def classify_sentences(sentences):
    """Return one label per sentence. Classifier by default; LLM only if configured."""
    if not sentences:
        return []

    model, metadata = load_classifier()
    labels = metadata["labels"]

    if llm.is_enabled():
        try:
            results = []
            for sentence in sentences:
                answer = llm.complete(LLM_PROMPT, sentence).strip().lower()
                match = next((l for l in labels if l in answer), None)
                results.append(match or str(model.predict([sentence])[0]))
            return results
        except llm.LLMError:
            pass  # fall through to the fitted classifier

    return [str(label) for label in model.predict(sentences)]


def extract_decisions(text):
    sentences = split_sentences(text)
    labels = classify_sentences(sentences)
    return [s for s, label in zip(sentences, labels) if label == "decision"]


def extract_action_items(text):
    sentences = split_sentences(text)
    labels = classify_sentences(sentences)
    return [
        ActionItem(
            task=sentence,
            owner=extract_owner(sentence),
            due_date=extract_due_date(sentence),
            source_text=sentence,
        )
        for sentence, label in zip(sentences, labels)
        if label == "action_item"
    ]


def extract_actions(text):
    return [item.task for item in extract_action_items(text)]


def summarize(text, decisions=None, action_items=None):
    """A factual summary of what was recorded, not a generated abstract.

    The previous version returned the transcript's first sentence, which is
    typically "morning everyone, thanks for joining".
    """
    decisions = decisions if decisions is not None else []
    action_items = action_items if action_items is not None else []

    if not text.strip():
        return "No transcript content was provided."
    if not decisions and not action_items:
        return "No decisions or action items were recorded in this meeting."

    parts = []
    if decisions:
        parts.append(f"{len(decisions)} decision{'s' if len(decisions) != 1 else ''} recorded")
    if action_items:
        owners = sorted({item.owner for item in action_items if item.owner})
        clause = f"{len(action_items)} action item{'s' if len(action_items) != 1 else ''}"
        if owners:
            clause += f" owned by {', '.join(owners)}"
        parts.append(clause)
    # Deliberately not str.capitalize(): it lowercases the rest of the string,
    # which mangles owner names ("owned by priya").
    sentence = "; ".join(parts)
    return sentence[:1].upper() + sentence[1:] + "."


def extract_meeting_intelligence(text):
    sentences = split_sentences(text)
    labels = classify_sentences(sentences)

    decisions = [s for s, label in zip(sentences, labels) if label == "decision"]
    action_items = [
        ActionItem(
            task=sentence,
            owner=extract_owner(sentence),
            due_date=extract_due_date(sentence),
            source_text=sentence,
        )
        for sentence, label in zip(sentences, labels)
        if label == "action_item"
    ]

    return MeetingSummary(
        summary=summarize(text, decisions, action_items),
        decisions=decisions,
        action_items=action_items,
    )


def classify_transcript_sentences(text):
    """Label for every sentence, including `neither`.

    Kept separate from `extract_meeting_intelligence` so the response schema does
    not change: attaching these to `MeetingSummary` would alter the shape the
    contract checks pin, for the sake of a monitoring signal.

    Drift monitoring needs the whole mix. Observing only extracted items would
    compare a live distribution that can never contain `neither` against a
    reference that is 58% `neither`, and report drift on every transcript.
    """
    return classify_sentences(split_sentences(text))
