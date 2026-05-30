
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nlp.extractor import extract_meeting_intelligence


df = pd.read_csv(ROOT / "datasets" / "sample_data.csv")
correct = 0
for row in df.to_dict("records"):
    result = extract_meeting_intelligence(row["transcript"])
    correct += len(result.action_items) >= int(row["expected_actions"]) and len(result.decisions) >= int(row["expected_decisions"])

print("records:", len(df))
print("structure_accuracy:", correct / len(df))
