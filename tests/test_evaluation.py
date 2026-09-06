"""
Tests for the evaluation arithmetic.

These matter more than they look. Everything else in the suite checks the
system's behaviour; this checks the numbers that will be quoted in a report,
and a metric that is quietly wrong is worse than no metric at all -- it gets
believed.

The two properties worth stating aloud, because both are choices rather than
conventions:

  * a deferral is NOT counted as an error. Refusing is a legitimate outcome
    for this system, and scoring it as a failure would reward a version that
    guesses.
  * sort accuracy is measured only over items the system CHOSE to sort. A
    system that defers everything has no sort accuracy, not a perfect one.

Run:  python -m tests.test_evaluation
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

from evaluation.metrics import Outcome, build_report, threshold_sweep
from tests.test_pipeline import PASSED, FAILED, check

logging.disable(logging.CRITICAL)


def sorted_ok(label="bottle", dest="RECYCLING", conf=0.95) -> Outcome:
    return Outcome(image="x.jpg", expected_label=label, expected_destination=dest,
                   detected=True, predicted_label=label, confidence=conf,
                   action="SORT", destination=dest, safety_rule="R7_APPROVED")


def sorted_wrong(label="bottle", expected="RECYCLING", landed="ORGANIC",
                 conf=0.90) -> Outcome:
    return Outcome(image="x.jpg", expected_label=label, expected_destination=expected,
                   detected=True, predicted_label=label, confidence=conf,
                   action="SORT", destination=landed, safety_rule="R7_APPROVED")


def deferred(label="cup", expected="MANUAL_CHECK", conf=0.80) -> Outcome:
    return Outcome(image="x.jpg", expected_label=label, expected_destination=expected,
                   detected=True, predicted_label=label, confidence=conf,
                   action="MANUAL_CHECK", destination="MANUAL_CHECK",
                   safety_rule="R4_UNCERTAIN")


def missed(label="battery", expected="HAZARDOUS") -> Outcome:
    return Outcome(image="x.jpg", expected_label=label, expected_destination=expected)


# ---------------------------------------------------------------------------

def test_verdicts():
    check("a correct sort is SORTED_CORRECT", sorted_ok().verdict == "SORTED_CORRECT")
    check("a wrong bin is SORTED_WRONG", sorted_wrong().verdict == "SORTED_WRONG")
    check("a manual check is DEFERRED", deferred().verdict == "DEFERRED")
    check("nothing detected is MISSED", missed().verdict == "MISSED")


def test_deferral_is_not_an_error():
    """The choice that shapes every other number here."""
    r = build_report([sorted_ok(), sorted_ok(), deferred(), deferred()])
    check("deferrals do not reduce sort accuracy",
          r.sort_accuracy == 1.0, f"{r.sort_accuracy}")
    check("but they do reduce coverage",
          r.coverage == 0.5, f"{r.coverage}")
    check("and are reported in their own right",
          r.deferral_rate == 0.5, f"{r.deferral_rate}")


def test_deferring_everything_is_not_perfect_accuracy():
    """The degenerate case a naive metric would score 100%."""
    r = build_report([deferred(), deferred(), deferred()])
    check("a system that never commits has zero sort accuracy, not one",
          r.sort_accuracy == 0.0, f"{r.sort_accuracy}")
    check("and zero coverage, which is what exposes it",
          r.coverage == 0.0, f"{r.coverage}")


def test_wrong_bin_is_counted_against_the_whole_set():
    r = build_report([sorted_ok(), sorted_ok(), sorted_ok(), sorted_wrong()])
    check("sort accuracy is 3 of 4", r.sort_accuracy == 0.75, f"{r.sort_accuracy}")
    check("wrong-bin rate is over everything presented",
          r.wrong_bin_rate == 0.25, f"{r.wrong_bin_rate}")


def test_misses_hurt_coverage_not_accuracy():
    r = build_report([sorted_ok(), sorted_ok(), missed(), missed()])
    check("an undetected object does not make the sorter look wrong",
          r.sort_accuracy == 1.0, f"{r.sort_accuracy}")
    check("it shows up as miss rate", r.miss_rate == 0.5, f"{r.miss_rate}")
    check("and as reduced coverage", r.coverage == 0.5, f"{r.coverage}")


def test_detection_and_system_are_scored_separately():
    """The interesting case: the detector is wrong, the system is right.

    A cup misread as a bowl still goes to manual check, because both are
    ambiguous across bins. The architecture absorbed a perception error, and
    the numbers should show that rather than hide it.
    """
    o = Outcome(image="x.jpg", expected_label="cup",
                expected_destination="MANUAL_CHECK",
                detected=True, predicted_label="bowl", confidence=0.88,
                action="MANUAL_CHECK", destination="MANUAL_CHECK",
                safety_rule="R4_UNCERTAIN")
    check("the object was named wrongly", not o.object_correct)
    check("but it still reached the right outcome", o.destination_correct)

    r = build_report([o, sorted_ok()])
    check("detection accuracy reflects the model's mistake",
          r.detection_accuracy == 0.5, f"{r.detection_accuracy}")
    check("while sort accuracy reflects the system's success",
          r.sort_accuracy == 1.0, f"{r.sort_accuracy}")


def test_per_class_breakdown():
    r = build_report([sorted_ok("bottle"), sorted_ok("bottle"),
                      sorted_wrong("bottle"), sorted_ok("banana", "ORGANIC")])
    bottle = r.by_class["bottle"]
    check("per-class counts are right",
          bottle["n"] == 3 and bottle["sorted_correct"] == 2
          and bottle["sorted_wrong"] == 1, str(bottle))
    check("per-class accuracy is computed over committed items only",
          abs(bottle["sort_accuracy"] - 2 / 3) < 0.001, str(bottle["sort_accuracy"]))
    check("classes are kept apart", r.by_class["banana"]["n"] == 1)


def test_confusion_records_where_things_landed():
    r = build_report([sorted_ok(), sorted_wrong(), missed("battery", "HAZARDOUS")])
    check("correct sorts sit on the diagonal",
          r.confusion["RECYCLING"]["RECYCLING"] == 1, str(r.confusion))
    check("a wrong bin is recorded against the expected row",
          r.confusion["RECYCLING"]["ORGANIC"] == 1, str(r.confusion))
    check("an undetected object is marked as such, not as a bin",
          r.confusion["HAZARDOUS"]["(not detected)"] == 1, str(r.confusion))


def test_threshold_sweep_shows_the_trade():
    """Higher threshold, fewer items, better accuracy -- the shape that makes
    the configured value defensible."""
    outcomes = ([sorted_ok(conf=0.95) for _ in range(6)]
                + [sorted_wrong(conf=0.45) for _ in range(4)])
    rows = {r["threshold"]: r for r in threshold_sweep(outcomes, [0.3, 0.9])}

    low, high = rows[0.3], rows[0.9]
    check("a low threshold covers everything",
          low["coverage"] == 1.0 and low["n_above"] == 10, str(low))
    check("and pays for it in accuracy",
          abs(low["accuracy"] - 0.6) < 0.001, str(low))
    check("a high threshold is accurate",
          high["accuracy"] == 1.0, str(high))
    check("but covers less of the stream",
          abs(high["coverage"] - 0.6) < 0.001, str(high))


def test_empty_report_does_not_divide_by_zero():
    r = build_report([])
    check("an empty set reports zeros rather than crashing",
          r.sort_accuracy == 0.0 and r.coverage == 0.0 and r.total == 0)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        print(f"\n{test.__name__}")
        test()
    print(f"\n{'-' * 60}\n{len(PASSED)} passed, {len(FAILED)} failed")
    for failure in FAILED:
        print(f"  FAILED: {failure}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
