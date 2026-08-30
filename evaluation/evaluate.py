"""Evaluate extraction on the held-out template split.

The previous version compared `len(result.action_items) >= expected_actions` and
called the result "structure_accuracy". Returning every sentence as an action
item would have scored 100%. These are real per-class precision, recall and F1,
plus slot-level scores, plus the keyword gate this replaced on the same data.
"""
import json
import sys
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nlp.extractor import extract_due_date, extract_owner, load_classifier  # noqa: E402
from training.train import LABELS, legacy_classify, load_sentences, score_slots, template_split  # noqa: E402


def main():
    model, metadata = load_classifier()
    rows = load_sentences()
    _, test_rows, held_out = template_split(rows)

    texts = [r["text"] for r in test_rows]
    gold = [r["label"] for r in test_rows]
    predicted = [str(p) for p in model.predict(texts)]
    legacy = [legacy_classify(t) for t in texts]

    print("Model :", metadata["model_type"])
    print("Data  :", metadata["data_source"], "(SYNTHETIC - not real transcripts)")
    print("Split :", metadata["split"])
    print(f"Held-out templates: {len(held_out)}  |  test sentences: {len(test_rows)}")
    print()

    print("=== sentence classification (learned) ===")
    print(classification_report(gold, predicted, zero_division=0))

    print("=== keyword gate this replaced, same data ===")
    print(classification_report(gold, legacy, zero_division=0))

    print(f"macro-F1  classifier {f1_score(gold, predicted, average='macro'):.4f}"
          f"   keyword gate {f1_score(gold, legacy, average='macro'):.4f}")
    print()

    matrix = confusion_matrix(gold, predicted, labels=LABELS)
    print("confusion matrix (rows=gold, cols=predicted)")
    width = max(len(l) for l in LABELS) + 2
    print(" " * width + "".join(f"{l:>14}" for l in LABELS))
    for name, row in zip(LABELS, matrix):
        print(f"{name:<{width}}" + "".join(f"{v:>14}" for v in row))
    print()

    print("=== slot extraction (rule-based, scored on true action items) ===")
    slots = score_slots(test_rows, gold)
    for slot, values in slots.items():
        print(f"{slot:10s} precision {values['precision']:.4f}  "
              f"recall {values['recall']:.4f}  f1 {values['f1']:.4f}  "
              f"(gold {values['gold_count']}, extracted {values['extracted_count']})")
    print()
    print("Owner recall is well below precision because the patterns anchor on")
    print("capitalised first names and miss full names, titles and team references")
    print('("the platform team", "Dr. Chen"). That is a real limitation of the rules,')
    print("surfaced by measuring them rather than assuming they work.")


if __name__ == "__main__":
    main()
