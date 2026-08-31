"""Fit the sentence classifier and measure the rule-based slot extractors.

Two things are evaluated, because two different mechanisms are at work:

1. **Sentence classification is learned** -- decision / action_item / neither.
   Split by TEMPLATE: whole phrasings are held out, so the test set is worded in
   ways the model has never seen. A row-level split would mostly measure
   memorisation of the template pool.
2. **Owner and due-date extraction is rule-based**, and is scored with precision
   and recall against gold slots. Rules are a reasonable choice for slot filling;
   asserting they work without measuring them was the problem.

The keyword gate this replaces is also scored, on the same held-out data, so the
improvement is a measured delta rather than an assertion.

    python training/train.py             # fit and write models/artifacts/
    python training/train.py --verify    # refit and fail if metrics drifted
"""
import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monitoring.drift import build_categorical_reference  # noqa: E402

from nlp.extractor import extract_due_date, extract_owner  # noqa: E402

DATA_PATH = ROOT / "datasets" / "transcripts.json"
ARTIFACT_DIR = ROOT / "models" / "artifacts"
DRIFT_REFERENCE_PATH = ARTIFACT_DIR / "drift_reference.json"
CLASSIFIER_PATH = ARTIFACT_DIR / "sentence_classifier.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.md"

RANDOM_STATE = 42
TEST_TEMPLATE_FRACTION = 0.25
C_GRID = [0.5, 1.0, 4.0, 10.0]
CV_FOLDS = 5
VERIFY_TOLERANCE = 0.05

LABELS = ["action_item", "decision", "neither"]

# The gate this replaces, kept so the comparison is measured rather than asserted.
LEGACY_ACTION_TERMS = ("action:", "follow up", "will", "todo")
LEGACY_DECISION_TERMS = ("decision:", "decided to", "approved")


def legacy_classify(sentence):
    lower = sentence.lower()
    if any(term in lower for term in LEGACY_DECISION_TERMS):
        return "decision"
    if any(term in lower for term in LEGACY_ACTION_TERMS):
        return "action_item"
    return "neither"


def load_sentences():
    meetings = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rows = []
    for meeting in meetings:
        for sentence in meeting["sentences"]:
            rows.append({**sentence, "meeting_id": meeting["meeting_id"]})
    return rows


def template_split(rows):
    """Hold out whole templates so the test set uses unseen phrasings."""
    rng = np.random.default_rng(RANDOM_STATE)
    held_out = set()
    for label in LABELS:
        templates = sorted({r["template_id"] for r in rows if r["label"] == label})
        n_hold = max(1, int(round(len(templates) * TEST_TEMPLATE_FRACTION)))
        held_out.update(rng.choice(templates, size=n_hold, replace=False).tolist())

    train = [r for r in rows if r["template_id"] not in held_out]
    test = [r for r in rows if r["template_id"] in held_out]
    return train, test, sorted(held_out)


def build_pipeline():
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2
            )),
        ])),
        ("model", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])


def score_slots(rows, predicted_labels):
    """Precision/recall of owner and due-date extraction on predicted action items.

    Scored only where the sentence really is an action item, so slot quality is
    not conflated with classification quality.
    """
    results = {}
    for slot, extractor in (("owner", extract_owner), ("due_date", extract_due_date)):
        correct = predicted = actual = 0
        for row, label in zip(rows, predicted_labels):
            gold = row.get(slot)
            found = extractor(row["text"]) if label == "action_item" else None
            if found is not None:
                predicted += 1
            if gold is not None:
                actual += 1
            if found is not None and gold is not None and str(found).lower() == str(gold).lower():
                correct += 1
        precision = correct / predicted if predicted else 0.0
        recall = correct / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results[slot] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "gold_count": actual,
            "extracted_count": predicted,
        }
    return results


def train():
    rows = load_sentences()
    train_rows, test_rows, held_out = template_split(rows)

    x_train = [r["text"] for r in train_rows]
    y_train = [r["label"] for r in train_rows]
    x_test = [r["text"] for r in test_rows]
    y_test = [r["label"] for r in test_rows]

    search = GridSearchCV(
        build_pipeline(), {"model__C": C_GRID}, scoring="f1_macro",
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_

    predicted = [str(p) for p in model.predict(x_test)]
    legacy = [legacy_classify(text) for text in x_test]

    per_class = {}
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predicted, labels=LABELS, zero_division=0
    )
    for index, label in enumerate(LABELS):
        per_class[label] = {
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1": round(float(f1[index]), 4),
            "support": int(support[index]),
        }

    legacy_per_class = {}
    lp, lr, lf, _ = precision_recall_fscore_support(
        y_test, legacy, labels=LABELS, zero_division=0
    )
    for index, label in enumerate(LABELS):
        legacy_per_class[label] = {
            "precision": round(float(lp[index]), 4),
            "recall": round(float(lr[index]), 4),
            "f1": round(float(lf[index]), 4),
        }

    metrics = {
        "model_type": "TF-IDF (word + char n-grams) -> LogisticRegression, 3-class",
        "data_source": "synthetic -- training/generate_transcripts.py",
        "labels": LABELS,
        "split": (
            "held-out TEMPLATES, not rows -- every test sentence uses phrasing "
            "absent from training"
        ),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "n_held_out_templates": len(held_out),
        "held_out_templates": held_out,
        "best_C": float(search.best_params_["model__C"]),
        "cv_macro_f1": round(float(search.best_score_), 4),
        "test_macro_f1": round(float(f1_score(y_test, predicted, average="macro")), 4),
        "per_class": per_class,
        "legacy_keyword_baseline": {
            "macro_f1": round(float(f1_score(y_test, legacy, average="macro")), 4),
            "per_class": legacy_per_class,
            "note": (
                "The keyword gate this replaced, scored on the same held-out data: "
                "'action:'/'follow up'/'will'/'todo' for actions, "
                "'decision:'/'decided to'/'approved' for decisions."
            ),
        },
        "slot_extraction": {
            "method": "rule-based patterns, measured against gold slots",
            **score_slots(test_rows, y_test),
        },
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    report = classification_report(y_test, predicted, zero_division=0)
    # Drift reference over the PREDICTED CLASS MIX across the whole corpus.
    # Class mix rather than confidence, for the reason recorded in
    # monitoring/drift.py: confidence on template-generated text is bimodal and
    # made the metric swing on ordinary traffic.
    drift_reference = build_categorical_reference(
        model.predict([r["text"] for r in rows])
    )
    return model, metrics, report, drift_reference


def render_model_card(metrics):
    legacy = metrics["legacy_keyword_baseline"]
    slots = metrics["slot_extraction"]
    rows = "\n".join(
        f"| `{label}` | {v['precision']} | {v['recall']} | {v['f1']} | {v['support']} | "
        f"{legacy['per_class'][label]['f1']} |"
        for label, v in metrics["per_class"].items()
    )
    return f"""# Model Card - Meeting Sentence Classification

## What this is

A three-class sentence classifier (decision / action_item / neither) feeding
rule-based slot extraction for owner and due date.

| Stage | Implementation |
|---|---|
| Sentence classification | **Learned** - TF-IDF (word 1-2, char_wb 3-5) -> LogisticRegression |
| Owner extraction | **Rules** - role-anchored patterns, measured |
| Due-date extraction | **Rules** - date expression patterns, measured |

Slot filling from an already-classified sentence is a reasonable use of patterns.
The problem with the previous version was not that it used rules -- it was that
nothing measured whether they worked.

## Training data - synthetic

Generated by `training/generate_transcripts.py` (seeded, reproducible). Not real
meeting transcripts. Only **5.9%** of decisions carry an explicit "Decision:"
marker; the rest are phrased the way decisions actually get recorded. Distractor
chatter shares vocabulary with real items ("I will grab a coffee", "we agreed the
weather is terrible") so a model keying on "will" or "agreed" is penalised.

## Split

**Held-out templates**, not rows: {metrics['n_held_out_templates']} whole phrasings are removed from
training, so every test sentence is worded in a way the classifier has never seen.

## Measured performance (held-out, n={metrics['n_test']})

Macro-F1 **{metrics['test_macro_f1']}** (cross-validated on train: {metrics['cv_macro_f1']}).

| Class | Precision | Recall | F1 | Support | Keyword-gate F1 |
|---|---|---|---|---|---|
{rows}

The final column is the keyword gate this replaced, scored on the same held-out
data: **macro-F1 {legacy['macro_f1']}** against the classifier's
**{metrics['test_macro_f1']}**. The improvement is measured, not asserted.

## Slot extraction (rule-based, on true action items)

| Slot | Precision | Recall | F1 | Gold |
|---|---|---|---|---|
| Owner | {slots['owner']['precision']} | {slots['owner']['recall']} | {slots['owner']['f1']} | {slots['owner']['gold_count']} |
| Due date | {slots['due_date']['precision']} | {slots['due_date']['recall']} | {slots['due_date']['f1']} | {slots['due_date']['gold_count']} |

## Known limitations

- Sentences are classified independently; no discourse context, so "that's
  approved then" is judged without knowing what "that" refers to.
- One label per sentence: a sentence recording a decision *and* assigning an
  owner gets one class.
- Due dates are extracted as surface strings ("Friday", "EOD") and never resolved
  to calendar dates, which needs the meeting date and a timezone.
- Owner extraction matches capitalised first names only -- no surname handling,
  no disambiguation between two people with the same first name, no mapping to
  directory identities.
- Templated synthetic text. Real transcripts carry ASR errors, interruptions,
  crosstalk and incomplete sentences; none appear here.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    model, metrics, report, drift_reference = train()

    if args.verify:
        if not METRICS_PATH.exists():
            print("FAIL: metrics.json missing; run without --verify first")
            return 1
        committed = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        old, new = committed["test_macro_f1"], metrics["test_macro_f1"]
        if abs(old - new) > VERIFY_TOLERANCE:
            print(f"FAIL: macro-F1 drifted {old} -> {new}")
            return 1
        print(f"OK: retrained macro-F1 {new} matches committed {old}")
        return 0

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CLASSIFIER_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    MODEL_CARD_PATH.write_text(render_model_card(metrics), encoding="utf-8")
    # The class mix the classifier produced at training time, so the running
    # service can tell whether transcripts have changed shape.
    DRIFT_REFERENCE_PATH.write_text(
        json.dumps(drift_reference, indent=2) + "\n", encoding="utf-8"
    )

    print(f"split          : {metrics['n_held_out_templates']} templates held out "
          f"({metrics['n_train']} train / {metrics['n_test']} test sentences)")
    print(f"classifier     : macro-F1 {metrics['test_macro_f1']} (C={metrics['best_C']})")
    print(f"keyword gate   : macro-F1 {metrics['legacy_keyword_baseline']['macro_f1']}")
    slots = metrics["slot_extraction"]
    print(f"owner slot     : P {slots['owner']['precision']} R {slots['owner']['recall']}")
    print(f"due_date slot  : P {slots['due_date']['precision']} R {slots['due_date']['recall']}")
    print()
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
