# Evaluation — the results chapter

Every other test in this repository asks *"given this detection, does the
right rule fire?"* — questions about logic, answered with scripted inputs.
They prove the system is **correct**. They say nothing about whether it
**works**.

This directory answers the question a reader of the report actually has.

```bash
python -m evaluation.capture --label bottle
```

```bash
python -m evaluation.evaluate
```

About ten minutes of photography, and the project has empirical results.

---

## Building the set

The folder name is the label, so there is nothing to annotate:

```
evaluation/dataset/
  bottle/001.jpg 002.jpg ...
  banana/...
  cell phone/...
```

That works because this system's output is a **decision per object**, not a
bounding box. One object per frame, and the folder says what it should have
been. No boxes to draw, no annotation tool to learn.

`capture.py` draws the detector's live output while you shoot, so you find
out immediately whether the model can see the object at all. Hold up a
battery and no box appears — that is the Stage 5 limitation, in front of you,
before you have wasted an hour on it. **Save the frame anyway:** a miss is a
real outcome and the evaluation counts it.

Aim for **20+ frames per class**, varying angle, distance and lighting. Below
10 the per-class rows are marked as indicative rather than measured.

Underscores are handled — `--label cell_phone` resolves to COCO's
`cell phone` and saves under the right name. That mismatch would otherwise
score every correctly-sorted phone as an error, silently.

---

## Reading the results

Four numbers, and the order matters.

**Sort accuracy** — of the items the system *chose to sort*, how many reached
the correct bin. This is the headline. Items sent to manual inspection are
not counted as errors here, because refusing is a legitimate outcome for this
architecture and scoring it as failure would reward a version that guesses.

**Coverage** — how much of the stream was handled without a human. This is
what stops sort accuracy being gamed. A system that defers everything has
100% of nothing; reported together, the pair is honest.

**Wrong-bin rate** — the costly error, as a share of everything presented.
Glass in the plastics stream contaminates a batch; this is the number an
operator cares about.

**Detection accuracy** — did the model name the object correctly. Reported
for comparison only, and the gap between this and sort accuracy is the
interesting part: a `cup` misread as a `bowl` still goes to manual check, so
the architecture absorbed a perception error. That gap is the argument for
having agents at all.

---

## The threshold table

The project has always been careful to call `confidence.auto_sort` an
operating threshold rather than a measured figure. The sweep at the bottom of
the report is how that disclaimer becomes a result:

```
 threshold  n above  accuracy  coverage
      0.50       48       79%       96%
      0.85       31       94%       62%
```

Read the *trade*, not a single row. A higher threshold sorts less and sorts
it better. Which value is right depends on whether a wrong bin or a queue at
the inspection station costs more — an operational decision, and this table
is what makes it arguable instead of arbitrary.

If your table shows accuracy *flat* across thresholds, that is a finding too:
it means confidence is not separating right from wrong on your data, and the
threshold is not earning its place.

---

## What gets written

`evaluation/results/evaluation.json` — the summary, per-class breakdown,
confusion matrix, rule breakdown, threshold sweep, and every individual
outcome. The reporting view picks it up automatically. It records which model
and which label map produced it, so results from different models never get
confused with one another.

Both `dataset/` and `results/` are gitignored. Photographs of your desk do
not belong in the repository.

---

## Re-run it after any change

The evaluation is the check that catches what unit tests cannot: a threshold
tuned badly, a label mapped wrongly, a new model that is worse than the old
one. Re-run it after changing `confidence.*`, after editing a label map, and
above all after swapping in a trained model — that is the moment the numbers
either justify Stage 5 or say it did not help.
