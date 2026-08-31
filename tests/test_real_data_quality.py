"""Quality gate for the real-data track (AMI).

The bars here are low because the honest result is low: on real meetings this
model finds decisions badly. The tests protect the *framing* -- that the headline
stays the positive-class figure and not the flattering three-class one -- as much
as the numbers.

Skips cleanly when the corpus is not cached, so a fresh clone stays green.

Observed at the time of writing:
    positive-class macro-F1  0.1799   keyword baseline 0.0244
    3-class macro-F1         0.4462   keyword baseline 0.3438
    decision F1 0.1417, action_item F1 0.2182, neither F1 0.9787
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.fetch_real_data import ARCHIVE  # noqa: E402

METRICS_PATH = ROOT / "models" / "artifacts" / "real" / "metrics.json"

MIN_POSITIVE_MACRO_F1 = 0.13
MIN_THREE_CLASS_MACRO_F1 = 0.38

pytestmark = pytest.mark.skipif(
    not ARCHIVE.exists() or not METRICS_PATH.exists(),
    reason="AMI not cached; run training/fetch_real_data.py then train_real.py",
)


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def test_headline_is_the_positive_class_figure(metrics):
    """The flattering number must not become the headline.

    Three-class macro-F1 averages in `neither` at ~0.98 on 98% of the data. A
    model scoring 0.14 on decisions lands at 0.45 that way, which reads far
    better than it deserves. The artifact must carry the positive-class figure
    and must say why.
    """
    assert "headline_positive_class_macro_f1" in metrics
    assert "carried by `neither`" in metrics["headline_note"]
    assert metrics["headline_positive_class_macro_f1"] < metrics["test_macro_f1"], (
        "the positive-class figure should be the lower, harder one"
    )


def test_it_still_beats_the_keyword_extractor(metrics):
    """The only performance claim made: better than what it replaced."""
    fitted = metrics["headline_positive_class_macro_f1"]
    baseline = metrics["keyword_baseline"]["positive_class_macro_f1"]
    assert fitted > baseline, "the fitted model no longer beats the keyword extractor"
    assert fitted >= MIN_POSITIVE_MACRO_F1
    assert metrics["test_macro_f1"] >= MIN_THREE_CLASS_MACRO_F1


def test_split_is_by_meeting(metrics):
    """Row-level splits leak: utterances share a meeting's topic and vocabulary."""
    assert "held-out MEETINGS" in metrics["split"]
    assert metrics["n_held_out_meetings"] > 0
    assert len(metrics["held_out_meetings"]) == metrics["n_held_out_meetings"]


def test_accuracy_is_never_reported(metrics):
    """At a 1.6% positive rate, accuracy is actively misleading."""
    assert "accuracy_note" in metrics
    for scores in metrics["per_class"].values():
        assert "accuracy" not in scores


def test_class_imbalance_is_disclosed(metrics):
    """The difficulty of this task is mostly its imbalance; hiding it hides that."""
    assert metrics["positive_rate"] < 0.05
    counts = metrics["class_counts"]
    assert counts["neither"] > 50 * (counts["decision"] + counts["action_item"])


def test_construction_choices_are_recorded(metrics):
    """Dropped data must be counted, not silently discarded."""
    stats = metrics["dataset_construction_stats"]
    assert "ambiguous_dropped" in stats
    assert "meetings_without_summlink" in stats
    assert "summlink" in metrics["label_construction"]


def test_real_data_is_attributed(metrics):
    """CC BY 4.0 requires attribution, so the artifact carries it."""
    assert "REAL" in metrics["data_source"]
    assert metrics["citation"]
    assert "groups.inf.ed.ac.uk/ami" in metrics["url"]
