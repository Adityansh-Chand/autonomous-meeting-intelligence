"""Behaviour of transcript extraction."""
import pytest

from nlp.chunker import chunk_transcript
from nlp.extractor import (
    extract_action_items,
    extract_decisions,
    extract_due_date,
    extract_meeting_intelligence,
    extract_owner,
    load_classifier,
    split_sentences,
    summarize,
)
from schema.output_schema import MeetingSummary, validate_output


def test_chunker_respects_the_size_budget():
    text = " ".join(f"Sentence number {i} in the transcript." for i in range(60))
    chunks = chunk_transcript(text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)


def test_sentence_splitting_strips_speaker_prefixes():
    sentences = split_sentences("Priya: we agreed to ship.\nArjun: sounds good.")
    assert sentences == ["we agreed to ship", "sounds good"]


def test_classifier_artifact_declares_its_labels():
    _, metadata = load_classifier()
    assert set(metadata["labels"]) == {"action_item", "decision", "neither"}


def test_decision_without_the_literal_marker_is_found():
    """The keyword gate required 'decision:' or 'decided to'; this has neither."""
    decisions = extract_decisions("Maya: we agreed to move forward with the beta launch.")
    assert decisions


def test_action_item_is_found_with_owner_and_date():
    items = extract_action_items("Priya will send the notes on the rollout by Friday.")
    assert items
    assert items[0].owner == "Priya"
    assert items[0].due_date == "Friday"


def test_owner_extraction_handles_several_phrasings():
    assert extract_owner("Priya will send the notes") == "Priya"
    assert extract_owner("can Arjun pick up the review") == "Arjun"
    assert extract_owner("that one's on Maya to finish") == "Maya"


def test_owner_extraction_returns_none_rather_than_guessing():
    """A wrong owner is worse than a missing one."""
    assert extract_owner("the platform team should look at this") is None


def test_due_date_extraction_covers_common_forms():
    assert extract_due_date("finish it by Friday") == "Friday"
    assert extract_due_date("due 2026-05-12") == "2026-05-12"
    assert extract_due_date("wrap up by EOD").lower() == "eod"
    assert extract_due_date("get it done next week") == "next week"


def test_due_date_absent_when_no_date_is_present():
    assert extract_due_date("someone should own this") is None


def test_summary_describes_what_was_recorded():
    """The old version returned the transcript's first sentence."""
    transcript = (
        "Ravi: morning everyone, thanks for joining.\n"
        "Maya: we agreed to move forward with the beta launch.\n"
        "Priya will send the notes on the rollout by Friday."
    )
    result = extract_meeting_intelligence(transcript)
    assert "morning everyone" not in result.summary
    assert "decision" in result.summary.lower() or "action" in result.summary.lower()


def test_empty_meeting_says_so_rather_than_inventing_content():
    result = extract_meeting_intelligence(
        "Ravi: morning everyone.\nMaya: sorry, could you repeat that."
    )
    assert isinstance(result, MeetingSummary)
    if not result.decisions and not result.action_items:
        assert "No decisions or action items" in result.summary


def test_no_transcript_content():
    assert "No transcript content" in summarize("")


def test_output_validates_against_the_schema():
    result = extract_meeting_intelligence(
        "Maya: we agreed to ship.\nPriya will send the notes by Friday."
    )
    payload = result.model_dump()
    assert validate_output(payload)
    assert set(payload) == {"summary", "decisions", "action_items"}


def test_action_items_carry_their_source_text():
    items = extract_action_items("Priya will send the notes by Friday.")
    assert items[0].source_text
