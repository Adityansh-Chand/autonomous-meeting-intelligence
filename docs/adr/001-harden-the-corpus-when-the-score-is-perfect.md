# ADR-001 — Treat a perfect score as a warning about the corpus

**Status:** Accepted · **Date:** 2026-06

## Context

Owner extraction is rule-based: patterns that pull a name or a team out of a sentence already
classified as an action item. Recall started at **0.3484** — the patterns anchored on a
single capitalised first name and missed full names, titles, teams and several assignment
phrasings.

Rewriting them around two owner shapes, person or team, with cue phrases enumerated, took
recall to **1.0000** on held-out templates.

That is the second time this repository has produced a perfect slot score. An earlier
version scored 1.0 / 1.0 and proved only that the patterns and the generator had been
written by the same hand.

The corpus is generated from about twenty templates. A thorough rule set will eventually
cover all of them, and covering them measures the corpus rather than the extractor.

## Decision

Treat a perfect score as evidence the task is too easy, and **make the task harder** rather
than banking the number.

Owner forms were added to the generator, chosen from how people actually name each other
rather than from what is awkward to parse:

- **`Chen and Maya`** — real work is shared, and a single-capture pattern cannot say so
- **`the on-call engineer`** — a role, not a name
- **`Priya's team`** — possessive delegation
- **`R. Raman`** — an initial, common in written notes

Reported figures against the harder corpus: **0.7465 / 0.6023 / 0.6667**.

**A test asserts both slot scores stay below 0.95**, so this has to be confronted rather
than rediscovered a third time.

## Alternatives considered

**Report the 1.0000.** It is a real measurement on a real held-out split. Rejected because
of what it measures: with the generator and the patterns written by the same author, a
perfect score says the author was consistent, not that the extractor works. The split being
held-out does not help when the *templates* are shared.

**Report 1.0000 with a caveat about corpus difficulty.** Rejected on the same reasoning as
the sales headline — the number a reader remembers is the number, and a caveat does not
survive being quoted.

**Weaken the patterns until the score drops.** Rejected as the same dishonesty in reverse. It
would produce a plausible number by making the code worse, and the extractor would then be
bad at a task it had already solved.

**Choose adversarial cases designed to defeat the patterns.** Rejected as the opposite
failure. The added forms had to be things that genuinely appear in meeting notes; picking
inputs specifically because the regex cannot parse them measures the author's imagination
rather than the task.

## Consequences

- Reported slot figures are lower and mean something. Still a large improvement on
  0.5745 / 0.3484 / 0.4337, now against a task that got harder at the same time.
- **The precision drop from 1.0000 is informative rather than a regression.** On
  *"Chen and Maya will…"* the extractor returns *"Chen"* — a partial answer counted as
  wrong. The extractor cannot represent a multi-owner action item, and the corpus now says
  so out loud instead of never asking.
- That limitation is a named remaining improvement, and fixing it is a schema change (the
  output field is singular), not a pattern change — which is why it is scoped as work rather
  than a bug.
- The `< 0.95` gate is unusual: a test that fails when results are *too good*. It is the
  only mechanism that reliably catches this, because every other signal points the wrong way.
- The same reflex caught a result in the other direction elsewhere in the portfolio, where
  BANKING77 scored higher than predicted and was verified for leakage rather than assumed
  broken. **Suspicion is symmetric or it is bias.**

## Revisit when

The generator is replaced by real annotated transcripts with slot labels. At that point the
corpus is not ours to make harder, and a high score becomes evidence again. AMI does not
close this — its summaries name owners in prose rather than in structured fields, which is
why slot extraction is still measured on synthetic data alone.
