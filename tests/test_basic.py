from nlp.chunker import chunk_transcript
from nlp.extractor import extract_actions, extract_meeting_intelligence
from schema.output_schema import MeetingSummary, validate_output


def test_extracts_action_items_with_owner_and_due_date():
    result = extract_meeting_intelligence(
        "Decision: approve launch. Priya will follow up with legal by Friday."
    )

    assert result.decisions == ["approve launch."]
    assert result.action_items[0].owner == "Priya"
    assert result.action_items[0].due_date == "Friday"


def test_extract_actions_keeps_backward_compatible_task_list():
    actions = extract_actions("Maya will send notes by tomorrow.")

    assert actions == ["Maya will send notes by tomorrow."]


def test_chunker_returns_non_empty_chunks():
    chunks = chunk_transcript("A short update. Another useful sentence.")

    assert chunks == ["A short update. Another useful sentence."]


def test_schema_validation():
    output = validate_output({
        "summary": "Launch sync",
        "decisions": ["Ship beta"],
        "action_items": [{"task": "Send notes"}],
    })

    assert isinstance(output, MeetingSummary)
