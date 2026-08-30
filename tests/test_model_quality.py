"""Quality gate for extraction, recomputed from the artifact.

Observed at the time of writing: classifier macro-F1 0.5894, keyword gate 0.3235,
owner slot P 0.5745 / R 0.3484, due-date slot P 1.0 / R 0.7160.

The bars look modest. They are measured on **held-out templates** -- every test
sentence is phrased in a way the classifier never saw -- which is a much harder
task than a row-level split, and the only one worth reporting.
"""
import sys
from pathlib import Path

import pytest
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nlp.extractor import load_classifier  # noqa: E402
from training.train import legacy_classify, load_sentences, score_slots, template_split  # noqa: E402

MIN_MACRO_F1 = 0.50
MIN_DECISION_RECALL = 0.60
MIN_ACTION_RECALL = 0.75
MIN_DUE_DATE_PRECISION = 0.85
# Three classes, so chance is ~0.33.
CHANCE_MACRO_F1 = 0.33


@pytest.fixture(scope="module")
def evaluated():
    model, metadata = load_classifier()
    rows = load_sentences()
    train_rows, test_rows, held_out = template_split(rows)
    texts = [r["text"] for r in test_rows]
    gold = [r["label"] for r in test_rows]
    predicted = [str(p) for p in model.predict(texts)]
    legacy = [legacy_classify(t) for t in texts]
    return metadata, train_rows, test_rows, gold, predicted, legacy, held_out


def test_classifier_meets_bar(evaluated):
    _, _, _, gold, predicted, _, _ = evaluated
    assert f1_score(gold, predicted, average="macro") >= MIN_MACRO_F1


def test_classifier_beats_chance(evaluated):
    _, _, _, gold, predicted, _, _ = evaluated
    assert f1_score(gold, predicted, average="macro") > CHANCE_MACRO_F1


def test_classifier_beats_the_keyword_gate_it_replaced(evaluated):
    """The central claim of this rebuild, asserted on held-out data."""
    _, _, _, gold, predicted, legacy, _ = evaluated
    assert f1_score(gold, predicted, average="macro") > f1_score(
        gold, legacy, average="macro"
    )


def test_decisions_without_explicit_markers_are_still_found(evaluated):
    """Only ~4% of decisions carry a 'Decision:' marker in this corpus."""
    _, _, test_rows, gold, predicted, _, _ = evaluated
    unmarked = [
        (g, p) for row, g, p in zip(test_rows, gold, predicted)
        if g == "decision" and not row["text"].startswith("Decision:")
    ]
    assert unmarked
    found = sum(1 for g, p in unmarked if p == "decision")
    assert found / len(unmarked) >= MIN_DECISION_RECALL


def test_action_item_recall_is_high(evaluated):
    """Missing a commitment is worse than surfacing a candidate for review."""
    _, _, _, gold, predicted, _, _ = evaluated
    actual = [(g, p) for g, p in zip(gold, predicted) if g == "action_item"]
    assert sum(1 for g, p in actual if p == "action_item") / len(actual) >= MIN_ACTION_RECALL


def test_due_date_extraction_is_precise(evaluated):
    """A wrong due date is worse than no due date, so precision is what is gated."""
    _, _, test_rows, gold, _, _, _ = evaluated
    slots = score_slots(test_rows, gold)
    assert slots["due_date"]["precision"] >= MIN_DUE_DATE_PRECISION


def test_slot_extraction_is_not_suspiciously_perfect(evaluated):
    """An earlier corpus scored owner and date at 1.0/1.0.

    That only demonstrated the patterns and the generator had been written by the
    same hand. The corpus now includes surface forms the rules do not handle.
    """
    _, _, test_rows, gold, _, _, _ = evaluated
    slots = score_slots(test_rows, gold)
    assert slots["owner"]["recall"] < 0.95
    assert slots["due_date"]["recall"] < 0.95


def test_split_holds_out_whole_templates(evaluated):
    _, train_rows, test_rows, _, _, _, held_out = evaluated
    train_templates = {r["template_id"] for r in train_rows}
    test_templates = {r["template_id"] for r in test_rows}
    assert train_templates.isdisjoint(test_templates)
    assert len(held_out) >= 5


def test_data_provenance_is_declared_as_synthetic(evaluated):
    metadata, _, _, _, _, _, _ = evaluated
    assert "synthetic" in metadata["data_source"].lower()
