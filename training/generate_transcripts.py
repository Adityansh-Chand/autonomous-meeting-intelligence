"""Generate labelled meeting transcripts with gold decisions and action items.

The three sample transcripts this replaces contained the literal tokens
`"Decision:"` and `"Action:"` because the extractor searched for exactly those
strings. Evaluation reported success by construction.

This generates transcripts a keyword matcher cannot solve:

- **Decisions are usually not announced.** Only a minority carry an explicit
  "Decision:" marker; the rest read as "we agreed to ship on the 12th", "the
  call landed on option B", "that's approved then".
- **Action items rarely say "action".** They appear as "Priya will send the
  notes", "can you chase legal on that", "that one's on Arjun before Friday".
- **Explicit negatives are included** -- "no decision was made today", "we
  parked that one" -- so a model that fires on every sentence is penalised.
- **Distractor chatter** carries the same vocabulary as real items without
  being one: "I will grab a coffee", "we agreed the weather is terrible".
- Owner names, date expressions and phrasings vary, and sentences are held out
  by TEMPLATE so the test set uses wording never seen in training.

Deterministic: fixed seed, no wall-clock reads.

    python training/generate_transcripts.py           # write datasets/transcripts.json
    python training/generate_transcripts.py --check   # fail if output would differ
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "datasets" / "transcripts.json"

SEED = 20260830
N_MEETINGS = 420

SPEAKERS = ["Priya", "Arjun", "Maya", "Ravi", "Chen", "Sofia", "Daniel", "Aisha",
            "Tomas", "Leila", "Marcus", "Nadia"]

# Owner surface forms the patterns do not all handle: full names, titles, and
# team references. Same reasoning as DATES above.
# Owner surface forms. The same reasoning as DATES below: the extraction patterns
# must not be able to cover every form, or the score measures the corpus rather
# than the extractor.
#
# The first set is what the patterns handle -- plain names, surnames, titles,
# teams. Improving those took owner recall from 0.3484 to 1.0000 on held-out
# templates, and a perfect score is exactly the signal this repository treats as a
# warning: it meant the corpus had been exhausted, not that the problem was
# solved.
#
# So harder forms were added afterwards, chosen from how people actually name each
# other in meetings rather than from what would be awkward to parse:
OWNER_FORMS = [
    "{first}", "{first}", "{first}", "{first}",
    "{first} Raman", "Dr. {first}", "the platform team", "someone from finance",
    # Two owners. Real work is shared, and a single-capture pattern cannot say so.
    "{first} and {second}",
    # A role rather than a name -- whoever is holding the rota that week.
    "the on-call engineer",
    # Possessive delegation: the team is the owner, named via a person.
    "{first}'s team",
    # An initial. Common in written notes, and not a name-shaped token.
    "{initial}. Raman",
]

TOPICS = [
    "the beta launch", "the pricing update", "the vendor contract",
    "the migration plan", "the hiring loop", "the security review",
    "the Q3 roadmap", "the support backlog", "the billing rewrite",
    "the onboarding flow", "the data retention policy", "the API deprecation",
]

# Deliberately includes forms the extraction patterns were NOT written for
# ("before the audit", "3rd of June", "once legal signs off"). Without these the
# rule-based extractor scores a perfect 1.0, which would only prove the patterns
# and the generator were written by the same hand.
DATES = [
    "Friday", "Monday", "Tuesday", "Wednesday", "Thursday", "tomorrow",
    "next week", "EOD", "end of week", "end of the month", "2026-05-12",
    "2026-06-02", "the 14th", "next Tuesday", "in two weeks",
    "3rd of June", "before the audit", "once legal signs off",
    "ahead of the board meeting", "by close of play", "the week after next",
]

# Decisions. Only DECISION_TEMPLATES[0] carries the explicit marker the old
# extractor searched for; the rest are how decisions actually get recorded.
DECISION_TEMPLATES = [
    "Decision: we go ahead with {topic}",
    "we agreed to move forward with {topic}",
    "the call landed on proceeding with {topic}",
    "that's approved then, {topic} goes ahead",
    "we settled on the second option for {topic}",
    "consensus was to postpone {topic} until next quarter",
    "we've decided against {topic} for now",
    "final answer on {topic} is yes",
    "we're going with the phased approach for {topic}",
    "it was resolved that {topic} ships this cycle",
    "the group signed off on {topic}",
    "we concluded {topic} should be handed to the platform team",
    "after discussion {topic} is a no",
    "everyone was aligned on pausing {topic}",
    "we're proceeding with {topic} as discussed",
    "the outcome is that {topic} gets deprioritised",
    "locking in {topic} for this sprint",
    "we've committed to {topic}",
    "that settles it, {topic} is happening",
    "the room agreed {topic} is the way forward",
    "we opted to keep {topic} as is",
    "call it: {topic} is approved",
    "we are not moving ahead with {topic}",
    "greenlighting {topic} today",
]

# Action items. Templates carry {owner} and optionally {date}.
ACTION_TEMPLATES = [
    ("Action: {owner} to write up {topic} by {date}", True),
    ("{owner} will send the notes on {topic} by {date}", True),
    ("{owner} is going to chase legal about {topic} before {date}", True),
    ("that one's on {owner} to finish by {date}", True),
    ("can {owner} pick up {topic} by {date}", True),
    ("{owner} takes {topic} and reports back on {date}", True),
    ("{owner} will follow up on {topic}", False),
    ("{owner} agreed to draft the summary for {topic}", False),
    ("we need {owner} to review {topic} before we move", False),
    ("{owner} owns {topic} from here", False),
    ("leaving {topic} with {owner} to sort out by {date}", True),
    ("{owner} said they would handle {topic}", False),
    ("assigning {topic} to {owner}, due {date}", True),
    ("{owner} to circulate the deck for {topic} by {date}", True),
    ("{owner} is picking up {topic} and will report on {date}", True),
    ("next step for {topic} sits with {owner}", False),
    ("{owner} please confirm {topic} by {date}", True),
    ("we need a draft of {topic} from {owner}", False),
    ("{owner} has the action on {topic}", False),
    ("put {topic} on {owner} for {date}", True),
    ("{owner} to unblock {topic} before {date}", True),
    ("follow-up on {topic} belongs to {owner}", False),
]

# Neither. Deliberately includes near-misses that share vocabulary with the
# positive classes -- a model keying on "will" or "agreed" fires on these.
NEITHER_TEMPLATES = [
    "morning everyone, thanks for joining",
    "no decision was made on {topic} today",
    "we parked {topic} for now, nothing agreed",
    "I will grab a coffee before the next call",
    "we agreed the weather has been terrible this week",
    "{owner} was on mute for the first few minutes",
    "just to recap where we got to last time on {topic}",
    "the numbers on {topic} are still being pulled together",
    "does anyone have context on {topic}",
    "{owner} will be out next week so timing is tight",
    "let's come back to {topic} when we have the data",
    "sorry, could you repeat that last point about {topic}",
    "the deck for {topic} is in the shared drive",
    "we ran out of time before reaching {topic}",
    "there was some debate about {topic} but nothing settled",
    "I think {owner} raised this on the last call too",
    "nothing was decided about {topic} in the end",
    "we will circle back to {topic} eventually",
    "{owner} will be presenting at the offsite, unrelated to this",
    "the agenda listed {topic} but we skipped it",
    "someone should probably look at {topic} at some point",
    "{owner} agreed it has been a long week",
    "no owner was identified for {topic}",
    "we discussed {topic} without reaching a conclusion",
]


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def build_sentence(rng, kind, template_index):
    """Return (text, label, owner, due_date, template_id)."""
    topic = _pick(rng, TOPICS)
    first = _pick(rng, SPEAKERS)
    # A distinct second name, so "Chen and Chen" cannot be generated.
    second = _pick(rng, [name for name in SPEAKERS if name != first])
    owner = _pick(rng, OWNER_FORMS).format(
        first=first, second=second, initial=first[0]
    )
    date = _pick(rng, DATES)

    if kind == "decision":
        template = DECISION_TEMPLATES[template_index]
        return (
            template.format(topic=topic), "decision", None, None,
            f"decision:{template_index:02d}",
        )

    if kind == "action_item":
        template, has_date = ACTION_TEMPLATES[template_index]
        text = template.format(owner=owner, topic=topic, date=date)
        return (
            text, "action_item", owner, date if has_date else None,
            f"action_item:{template_index:02d}",
        )

    template = NEITHER_TEMPLATES[template_index]
    return (
        template.format(topic=topic, owner=owner), "neither", None, None,
        f"neither:{template_index:02d}",
    )


def generate(seed: int = SEED, n: int = N_MEETINGS):
    rng = np.random.default_rng(seed)
    meetings = []

    for index in range(n):
        n_decisions = int(rng.integers(0, 3))
        n_actions = int(rng.integers(0, 4))
        n_neither = int(rng.integers(3, 8))

        sentences = []
        for _ in range(n_decisions):
            sentences.append(build_sentence(
                rng, "decision", int(rng.integers(len(DECISION_TEMPLATES)))
            ))
        for _ in range(n_actions):
            sentences.append(build_sentence(
                rng, "action_item", int(rng.integers(len(ACTION_TEMPLATES)))
            ))
        for _ in range(n_neither):
            sentences.append(build_sentence(
                rng, "neither", int(rng.integers(len(NEITHER_TEMPLATES)))
            ))

        order = rng.permutation(len(sentences))
        sentences = [sentences[i] for i in order]

        lines, labels = [], []
        for text, label, owner, due_date, template_id in sentences:
            speaker = _pick(rng, SPEAKERS)
            lines.append(f"{speaker}: {text}.")
            labels.append({
                "text": text,
                "label": label,
                "owner": owner,
                "due_date": due_date,
                "template_id": template_id,
            })

        meetings.append({
            "meeting_id": f"mtg_{index:04d}",
            "tenant_id": f"tenant_{index % 23:03d}",
            "title": f"Sync {index:04d}",
            "participants": sorted({line.split(":")[0] for line in lines}),
            "transcript": "\n".join(lines),
            "source": _pick(rng, ["zoom", "teams", "meet", "uploaded_file"]),
            "sentences": labels,
            "gold_decisions": [s["text"] for s in labels if s["label"] == "decision"],
            "gold_action_items": [
                {"task": s["text"], "owner": s["owner"], "due_date": s["due_date"]}
                for s in labels if s["label"] == "action_item"
            ],
        })

    return meetings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    meetings = generate()
    text = json.dumps(meetings, indent=2) + "\n"

    if args.check:
        if not OUT_PATH.exists():
            print(f"FAIL: {OUT_PATH} is missing; run without --check first")
            return 1
        if OUT_PATH.read_text(encoding="utf-8") != text:
            print("FAIL: regenerated transcripts differ from the committed file")
            return 1
        print(f"OK: {OUT_PATH.name} is reproducible ({len(meetings)} meetings)")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")

    counts = {"decision": 0, "action_item": 0, "neither": 0}
    for meeting in meetings:
        for sentence in meeting["sentences"]:
            counts[sentence["label"]] += 1
    explicit = sum(
        1 for m in meetings for s in m["sentences"]
        if s["label"] == "decision" and s["text"].startswith("Decision:")
    )
    print(f"wrote {OUT_PATH} meetings={len(meetings)}")
    print(f"  sentences: {counts}")
    print(f"  decisions carrying an explicit 'Decision:' marker: "
          f"{explicit}/{counts['decision']} "
          f"({explicit / max(counts['decision'], 1):.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
