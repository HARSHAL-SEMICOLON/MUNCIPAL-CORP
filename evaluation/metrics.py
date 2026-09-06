"""
Turning a pile of decisions into a results chapter.

This module is deliberately pure -- it takes a list of finished Outcome
records and returns numbers. No camera, no model, no database. That is what
lets the arithmetic be tested (tests/test_evaluation.py) rather than trusted,
which matters more here than anywhere else in the project: these are the
figures that will be quoted in a report.

**The distinction this file exists to make.** A detection metric asks "did
YOLO name the object correctly?" That is a question about a model somebody
else trained. The question worth reporting is "did the *system* put the item
in the right bin, and when it was not sure, did it say so?" Those come apart
constantly -- the detector can be wrong about the object and the system still
right about the bin (a `cup` misread as a `bowl` is manual-checked either
way), and the detector can be right while the system defers.

So every outcome carries both, and the headline number is the one with
consequences:

    sort accuracy  = of the items the system CHOSE to sort, how many
                     went to the correct bin

An item sent to manual inspection is not counted as an error. Refusing is a
legitimate outcome for this system, and scoring it as a failure would reward
a version that guesses.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Outcome:
    """One image, start to finish."""
    image: str
    expected_label: str
    expected_destination: str

    detected: bool = False
    predicted_label: str = ""
    confidence: float = 0.0
    material: str = ""
    category: str = ""
    action: str = ""
    destination: str = ""
    safety_rule: str = ""
    reason: str = ""

    # -- derived -----------------------------------------------------------

    @property
    def object_correct(self) -> bool:
        return self.detected and self.predicted_label == self.expected_label

    @property
    def sorted_(self) -> bool:
        """Did the system commit to a bin, rather than deferring or holding?"""
        return self.action == "SORT"

    @property
    def destination_correct(self) -> bool:
        return self.destination == self.expected_destination

    @property
    def verdict(self) -> str:
        if not self.detected:
            return "MISSED"
        if self.action == "MANUAL_CHECK":
            return "DEFERRED"
        if self.action != "SORT":
            return "HELD"
        return "SORTED_CORRECT" if self.destination_correct else "SORTED_WRONG"


@dataclass
class Report:
    total: int = 0
    missed: int = 0
    deferred: int = 0
    held: int = 0
    sorted_correct: int = 0
    sorted_wrong: int = 0

    by_class: dict = field(default_factory=dict)
    confusion: dict = field(default_factory=dict)
    by_rule: Counter = field(default_factory=Counter)
    confidence_bands: list = field(default_factory=list)

    # -- headline figures --------------------------------------------------

    @property
    def committed(self) -> int:
        """Items the system chose to sort."""
        return self.sorted_correct + self.sorted_wrong

    @property
    def sort_accuracy(self) -> float:
        """THE number. Of what it sorted, how much went to the right bin."""
        return self.sorted_correct / self.committed if self.committed else 0.0

    @property
    def coverage(self) -> float:
        """How much of the stream it handled without a human."""
        return self.committed / self.total if self.total else 0.0

    @property
    def wrong_bin_rate(self) -> float:
        """The costly error, as a share of everything presented."""
        return self.sorted_wrong / self.total if self.total else 0.0

    @property
    def deferral_rate(self) -> float:
        return self.deferred / self.total if self.total else 0.0

    @property
    def miss_rate(self) -> float:
        return self.missed / self.total if self.total else 0.0

    @property
    def detection_accuracy(self) -> float:
        """For comparison only -- did the detector name the object right."""
        seen = sum(c["object_correct"] for c in self.by_class.values())
        return seen / self.total if self.total else 0.0

    def summary(self) -> dict:
        return {
            "images": self.total,
            "sort_accuracy": round(self.sort_accuracy, 4),
            "coverage": round(self.coverage, 4),
            "wrong_bin_rate": round(self.wrong_bin_rate, 4),
            "deferral_rate": round(self.deferral_rate, 4),
            "miss_rate": round(self.miss_rate, 4),
            "detection_accuracy": round(self.detection_accuracy, 4),
            "sorted_correct": self.sorted_correct,
            "sorted_wrong": self.sorted_wrong,
            "deferred": self.deferred,
            "held": self.held,
            "missed": self.missed,
        }


def build_report(outcomes: Iterable[Outcome]) -> Report:
    outcomes = list(outcomes)
    r = Report(total=len(outcomes))

    per_class: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "object_correct": 0, "sorted_correct": 0,
                 "sorted_wrong": 0, "deferred": 0, "missed": 0,
                 "confidence_sum": 0.0})
    confusion: dict[str, Counter] = defaultdict(Counter)

    for o in outcomes:
        verdict = o.verdict
        if verdict == "MISSED":
            r.missed += 1
        elif verdict == "DEFERRED":
            r.deferred += 1
        elif verdict == "HELD":
            r.held += 1
        elif verdict == "SORTED_CORRECT":
            r.sorted_correct += 1
        else:
            r.sorted_wrong += 1

        c = per_class[o.expected_label]
        c["n"] += 1
        c["confidence_sum"] += o.confidence
        if o.object_correct:
            c["object_correct"] += 1
        if verdict == "SORTED_CORRECT":
            c["sorted_correct"] += 1
        elif verdict == "SORTED_WRONG":
            c["sorted_wrong"] += 1
        elif verdict == "DEFERRED":
            c["deferred"] += 1
        elif verdict == "MISSED":
            c["missed"] += 1

        if o.safety_rule:
            r.by_rule[o.safety_rule] += 1

        # Rows are the expected bin, columns where it actually went. A
        # perfect system is diagonal; everything off it is a story.
        landed = o.destination if o.detected else "(not detected)"
        confusion[o.expected_destination][landed] += 1

    for label, c in per_class.items():
        n = c["n"]
        c["avg_confidence"] = round(c["confidence_sum"] / n, 4) if n else 0.0
        c.pop("confidence_sum")
        committed = c["sorted_correct"] + c["sorted_wrong"]
        c["sort_accuracy"] = round(c["sorted_correct"] / committed, 4) if committed else None
        c["deferral_rate"] = round(c["deferred"] / n, 4) if n else 0.0
    r.by_class = dict(per_class)
    r.confusion = {k: dict(v) for k, v in confusion.items()}
    r.confidence_bands = threshold_sweep(outcomes)
    return r


def threshold_sweep(outcomes: Iterable[Outcome],
                    steps: Iterable[float] | None = None) -> list[dict]:
    """What the confidence threshold actually buys.

    The project has always been careful to call 0.85 an operating threshold
    rather than a measured figure. This is how that claim stops being a
    disclaimer and becomes a result: for each candidate threshold, how
    accurate are the items above it, and how much of the stream do they
    cover?

    The trade is the point. A high threshold sorts almost nothing but sorts
    it correctly; a low one handles everything and puts glass in the plastics
    stream. The right value is an operational choice about which error costs
    more, and this table is what makes that choice arguable instead of
    arbitrary.
    """
    outcomes = [o for o in outcomes if o.detected]
    steps = steps if steps is not None else [i / 20 for i in range(6, 20)]
    rows = []
    for t in steps:
        above = [o for o in outcomes if o.confidence >= t]
        correct = sum(1 for o in above if o.destination_correct)
        rows.append({
            "threshold": round(t, 3),
            "n_above": len(above),
            "accuracy": round(correct / len(above), 4) if above else None,
            "coverage": round(len(above) / len(outcomes), 4) if outcomes else 0.0,
        })
    return rows
