# ADR-002 — Headline the positive classes, not the three-class average

**Status:** Accepted · **Date:** 2026-07

## Context

Validated on the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) — 137 real
recorded meetings, with decisions and action items linked to utterances through AMI's own
annotation layers.

In a real meeting **1.6%** of utterances carry a decision or an action item. The remaining
98% are `neither`, and `neither` is trivially easy: the classifier scores **0.9787** on it.

That produces two defensible numbers for the same model:

| | Fitted | Keyword extractor |
|---|---|---|
| **Positive-class macro-F1** | **0.1799** | 0.0244 |
| 3-class macro-F1 | 0.4462 | 0.3438 |

0.4462 is a real macro-F1 over the real label set. It is also mostly a measurement of how
easy `neither` is.

## Decision

Headline **0.1799**, the macro-F1 over `decision` and `action_item` only. Report 0.4462
alongside, explicitly labelled as not the headline and explaining why.

**A test asserts the positive-class figure stays the headline.**

## Alternatives considered

**Headline 0.4462, since three-class macro-F1 is the standard metric for a three-class
problem.** It is standard, and here it is inflated by construction: averaging in a class
that is 98% of the data and scores 0.9787 lifts a model scoring 0.1417 on decisions into
respectable-looking territory. The metric is not wrong; it answers a question nobody has.
Nobody deploys this to identify non-decisions.

**Report accuracy.** Worse still — predicting `neither` for every utterance scores 98.4%.
The same reasoning as the incident platform's base-rate problem, arriving from a different
domain.

**Drop `neither` from training as well as from the metric.** Rejected: the classifier must
handle the 98% at inference, so removing it from training would produce a model that
believes every utterance is a decision. The class belongs in training and does not belong in
the headline.

**Report only 0.1799 and omit the three-class figure.** Rejected. Someone will compute it,
and finding an unreported higher number looks like concealment. Publishing it labelled is
both more honest and more robust.

## Consequences

- **The headline is 0.1799, which is a weak absolute result, and the README says so
  plainly**: the model recovers about 21% of decisions and is right about 11% of the time
  when it flags one.
- The claim that survives is comparative and supported: **roughly 7× the keyword extractor**
  (0.1799 against 0.0244) on the classes that matter.
- The gap from 0.5894 on synthetic data is explained rather than hidden. In the generated
  corpus a decision is one sentence built from a decision template and the classes are
  near-balanced; in a real meeting the sentence that settles something is lexically
  indistinguishable from the twenty around it. A sparse bag of n-grams has little to work
  with — a limit of the representation, and closing it means sentence embeddings with
  surrounding context, not tuning this model.
- Comparison with published AMI results is harder, since conventions vary. Accepted: a
  metric chosen for comparability that misdescribes the model is a bad trade.

## Revisit when

The representation changes — contextual embeddings over surrounding utterances. The metric
choice stays correct regardless; what changes is whether the headline number stops being
embarrassing.
