# ADR-003 — Classify sentences, but fill slots with rules

**Status:** Accepted · **Date:** 2026-04

## Context

Extraction here has two stages: decide whether a sentence is a decision, an action item, or
neither; then, for action items, pull out the owner and the due date.

The original code did both with keywords — grep for the literal strings `"Decision:"`,
`"will"` and `"follow up"` — and the sample transcripts contained those exact strings,
because they were written to. The evaluation could not fail.

The instinct when replacing that is to learn both stages. It is the right answer for one of
them.

## Decision

**Learn the classification, keep the slots rule-based, and measure both.**

A TF-IDF (word + character n-gram) → logistic regression 3-class sentence classifier replaces
the keyword gate. Owner and due-date extraction stay pattern-based — and are now scored
against gold slots (precision / recall / F1) rather than assumed correct.

## Alternatives considered

**Learn the slots too, as sequence labelling (BIO tagging, a CRF, or a fine-tuned token
classifier).** The standard answer, and the one a reviewer expects. Rejected on the data
available: token-level span labels would have to come from the same generator that produces
the templates, so a sequence model would be learning to recover spans we placed. That is the
failure ADR-001 exists to prevent, arriving through a more sophisticated door.

Real span labels would change this. AMI does not supply them — its summaries name owners in
prose rather than in structured fields — which is precisely why slot extraction is still
measured on synthetic data alone, and why that is stated in the model card rather than
skated over.

**Keep keywords for classification too.** The status quo. Rejected, and measured: the keyword
gate scores **0.3235** against the classifier's **0.5894** on held-out synthetic data, and
**0.0244** against **0.1799** on AMI. A test asserts the classifier wins.

**Rules for both stages, better written.** Rejected for classification specifically: deciding
whether a sentence settles something is a judgement about meaning, and enumerable cues do not
cover it. Real transcripts phrase decisions without the token `"Decision:"` — which is the
whole reason the corpus generates them that way.

**An LLM for both.** Available behind the provider-agnostic seam, deliberately not the
default, and no reported metric uses it.

## Consequences

- The learned/written line is drawn on a principle that transfers: **learn what is a
  judgement, write what is a pattern.** Whether a sentence settles something is a judgement;
  the shape of a date is not.
- Rules are now *measured* rather than assumed, which is the substantive change. `1.0000`
  precision on due dates is a real number against a real gold set, and it is precisely why
  the corpus had to be made harder (ADR-001).
- Rules do not generalise beyond enumerated forms, and the reported owner recall of 0.6023
  is that limitation quantified rather than described.
- The classifier and the slot extractor fail independently, so a correctly classified action
  item can still yield the wrong owner. Both levels are reported, and the slot figures are
  scored **on true action items** so a classification error does not silently become a slot
  error.
- The same learn-perception / write-mechanism split appears in the customer operations
  service for
  [policy](https://github.com/Adityansh-Chand/ai-proactive-customer-operations/blob/main/docs/adr/001-deterministic-policy-over-learned.md).
  Two services reaching it independently is weak evidence it is the right line.

## Revisit when

Real span-annotated transcripts become available. That removes the objection above entirely
and makes sequence labelling an empirical question worth measuring rather than a structural
one worth avoiding.
