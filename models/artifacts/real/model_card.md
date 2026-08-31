# Model Card - Decision and Action Extraction on REAL meetings

## What this is

The **same pipeline** the service serves -- TF-IDF over word and character
n-grams into a logistic regression -- fitted on real recorded meetings instead of
generated transcripts.

**Data:** REAL -- AMI Meeting Corpus manual annotations (CC BY 4.0)
Carletta et al., 'The AMI Meeting Corpus: A Pre-Announcement', MLMI 2005
https://groups.inf.ed.ac.uk/ami/corpus/

### How the labels were built

AMI does not ship this dataset; it ships the parts. A dialogue act linked by extractive/summlink.xml to a summary sentence under decisions is a decision, under actions an action item, everything else neither -- labels derive from human annotators summarising the meeting, not from anyone seeking a learnable pattern.

That provenance is the point. The synthetic corpus is labelled by the same process
that generated it. Here the labels come from annotators writing a human summary of
a meeting under fixed headings, with no classifier in mind.

## This task is much harder than the synthetic one

| | Synthetic | Real (AMI) |
|---|---|---|
| Positive rate | roughly balanced | **0.0159** |
| Split unit | held-out templates | held-out meetings |
| Decisions | by construction, one per template | 1350 across 137 meetings |
| Action items | by construction | 371 |

108338 utterances, of which 1350 are
decisions and 371 are action items. In a real meeting
almost everything said is neither, and the sentence that settles a decision looks
much like the twenty around it.

## Measured performance (held-out meetings, n=29930)

### The honest headline: it finds real decisions badly

**Positive-class macro-F1 0.1799** --
the mean of `decision` and `action_item` F1, against
**0.0244** for the keyword
extractor it replaced.

| Class | Precision | Recall | F1 (fitted) | F1 (keyword baseline) | Support |
|---|---|---|---|---|---|
| `action_item` | 0.1435 | 0.4545 | **0.2182** | 0.0358 | 66 |
| `decision` | 0.1061 | 0.213 | **0.1417** | 0.013 | 399 |
| `neither` | 0.988 | 0.9697 | **0.9787** | 0.9825 | 29465 |

The three-class macro-F1 is **0.4462** against
0.3438 for the baseline, and that is the number
this repository declines to lead with. It averages in `neither` at
0.9787 -- a class that is 98% of the data and trivially easy --
which lifts a model scoring 0.1417 on decisions into
respectable-looking territory.

On the two classes anyone actually wants, this model is weak: it recovers about
21% of decisions, and when it flags one it
is right about 10% of the time. It does
beat the keyword extractor, which is the claim being made and the only one the
data supports.

Cross-validated macro-F1 on the training meetings:
0.4899.

**Accuracy is not reported.** At a 1.6% positive rate, predicting 'neither' everywhere scores 98.4% and finds nothing.

### Why this is so much lower than the synthetic 0.5894

Both numbers are real measurements; they measure different difficulties. In the
generated corpus a decision is one sentence built from a decision template, and
the classes are near-balanced. In a real meeting a decision emerges over several
turns of ordinary talk, 1.6% of utterances carry one, and the sentence that
settles it is lexically indistinguishable from the twenty around it.

A sparse bag of n-grams has very little to work with there. That is a limit of the
representation, not a bug, and closing it means sentence embeddings with
surrounding context -- not tuning this model.

## Why the split is by meeting

Two utterances from the same meeting share topic, participants and vocabulary. A
row-level split would let the model recognise the *meeting* rather than the
decision, in the same way the synthetic track's first version let it recognise
phrasings it had memorised. Held-out meetings is the equivalent guard for this
corpus.

34 of 137 meetings are held out,
chosen deterministically by position in the sorted meeting list so the split is
reproducible from the data alone.

## Construction notes

- 10 utterances were
  linked to both a decision and an action item. Real, but not representable in a
  3-class scheme, so they are dropped and counted rather than assigned to whichever
  class we would prefer.
- 5 summarised
  meetings ship no `summlink` file. Every utterance in them would default to
  `neither`, adding thousands of false negatives, so they are excluded.

## How this relates to the served model

The served extractor is trained on the synthetic corpus and this one is not
deployed. What transfers is the method, measured against meetings nobody wrote for
a classifier.

## Known limitations

- One corpus, one genre: scenario-driven design meetings among four participants.
  Real corporate meetings differ in length, turn-taking and vocabulary.
- Labels are utterance-level, but a decision is often spread across several turns;
  the linked act is the one an annotator considered supporting evidence, which is
  a narrower thing than "the decision".
- No slot extraction on this track: AMI's summaries name owners in prose rather
  than in a structured field, so owner and due-date accuracy cannot be measured
  here the way it is on the synthetic track.
