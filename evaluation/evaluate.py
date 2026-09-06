"""
Run the whole system over a labelled set and report what it actually did.

    python -m evaluation.evaluate

This is the project's results chapter. Every other test in the repository
asks "given this detection, does the right rule fire?" -- questions about
logic, answered with scripted inputs. This one asks the question a reader of
the report actually has: **does it work, and how well?**

It drives the real detector over real photographs and the real agents over
those detections. Nothing is scripted. The only thing supplied is the ground
truth, which is the folder each image sits in.

Two things it is careful about:

**Deferrals are not errors.** An item sent to manual inspection is the
system working, not failing. Scoring it as a miss would reward a version
that guesses, which is the opposite of what this architecture is for. So the
headline figure is accuracy *among the items it chose to sort*, reported
alongside how much of the stream that was.

**The detector is judged separately from the system.** They come apart, and
the difference is the interesting part -- a misread object that still lands
in the right bin says the architecture absorbed a perception error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging

import cv2

from agents.classification_agent import ClassificationAgent
from agents.decision_agent import DecisionAgent
from agents.material_agent import MaterialAgent
from agents.safety_agent import SafetyAgent, SafetyContext
from core.config import Config
from core.event_bus import EventBus
from core.labels import load_label_map
from core.routing import destination_for
from evaluation.labels import describe, resolve
from evaluation.metrics import Outcome, build_report
from vision.detector import Detector, ModelUnavailable

DATASET = ROOT / "evaluation" / "dataset"
RESULTS = ROOT / "evaluation" / "results"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the system on a labelled set")
    p.add_argument("--dataset", default=None, help="defaults to evaluation/dataset")
    p.add_argument("--config", default=None)
    p.add_argument("--min-per-class", type=int, default=10,
                   help="warn below this many images in a class")
    p.add_argument("--quiet", action="store_true", help="suppress per-image lines")
    return p.parse_args(argv)


class Harness:
    """The real agents, wired for one image at a time.

    Deliberately not the Pipeline: there is no belt, no tracking and no
    decision line here, because a photograph has no time dimension. What is
    under test is the reasoning chain -- material, category, decision,
    safety -- against a real detection.
    """

    def __init__(self, cfg: Config):
        bus = EventBus()                      # nothing subscribes; keeps agents quiet
        labels = load_label_map(cfg)
        self.labels = labels
        self.detector = Detector(cfg)
        self.material = MaterialAgent(bus, labels=labels)
        self.classifier = ClassificationAgent(bus, labels=labels)
        self.decider = DecisionAgent(cfg, bus)
        self.safety = SafetyAgent(cfg, bus)

    def resolve(self, folder: str) -> str:
        """Folder name -> a label the map knows, or the raw name if unmapped."""
        return resolve(folder, self.labels) or folder.strip().lower()

    def expected_destination(self, label: str) -> str:
        """Where an object of this class SHOULD end up, per the system's own
        rules. Derived, not hand-written, so ground truth cannot drift away
        from the configuration it is judging."""
        from agents.classification_agent import MATERIAL_TO_CATEGORY
        from core.messages import Category

        override = self.labels.override_for(label)
        if override is not None:
            return destination_for(override).value

        material = self.labels.material_for(label)
        if material is not None:
            cat = MATERIAL_TO_CATEGORY.get(material, Category.MANUAL_CHECK)
            return destination_for(cat).value

        ambiguity = self.labels.ambiguity_for(label)
        if ambiguity is not None:
            bins = {destination_for(MATERIAL_TO_CATEGORY.get(c, Category.MANUAL_CHECK))
                    for c in ambiguity.candidates}
            if len(bins) == 1:
                return bins.pop().value
        return destination_for(Category.MANUAL_CHECK).value

    def run_image(self, path: Path, expected_label: str) -> Outcome:
        expected_bin = self.expected_destination(expected_label)
        outcome = Outcome(image=path.name, expected_label=expected_label,
                          expected_destination=expected_bin)

        frame = cv2.imread(str(path))
        if frame is None:
            outcome.reason = "unreadable image"
            return outcome

        detections = self.detector.detect(frame, frame_id=1)
        if not detections:
            outcome.reason = "no detection"
            return outcome

        # One object per frame by construction, so the strongest detection is
        # the subject. Ties go to the larger box.
        det = max(detections, key=lambda d: (d.confidence, d.area))
        outcome.detected = True
        outcome.predicted_label = det.label
        outcome.confidence = det.confidence

        material = self.material.identify(det)
        classification = self.classifier.classify(det, material)
        decision = self.decider.decide(det, material, classification)
        verdict = self.safety.review(det, classification, decision,
                                     SafetyContext(neighbours=detections))

        outcome.material = material.material.value
        outcome.category = classification.category.value
        outcome.action = verdict.action.value
        outcome.destination = verdict.destination.value
        outcome.safety_rule = verdict.rule
        outcome.reason = verdict.reason
        return outcome


def load_dataset(root: Path) -> dict[str, list[Path]]:
    if not root.exists():
        return {}
    out = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        images = sorted(p for p in d.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if images:
            out[d.name.lower()] = images
    return out


def bar(value: float, width: int = 22) -> str:
    filled = int(round(value * width))
    return "#" * filled + "." * (width - filled)


def print_report(report, args) -> None:
    s = report.summary()
    print("\n" + "=" * 78)
    print("  EVALUATION RESULTS")
    print("=" * 78)
    print(f"\n  {s['images']} images\n")

    print(f"  sort accuracy       {report.sort_accuracy:>7.1%}   "
          f"{bar(report.sort_accuracy)}")
    print(f"    of the {report.committed} items the system chose to sort,")
    print(f"    {report.sorted_correct} went to the correct bin"
          f" and {report.sorted_wrong} did not")
    print()
    print(f"  coverage            {report.coverage:>7.1%}   {bar(report.coverage)}")
    print(f"    handled without a human")
    print()
    print(f"  wrong-bin rate      {report.wrong_bin_rate:>7.1%}   "
          f"{bar(report.wrong_bin_rate)}")
    print(f"    the costly error, as a share of everything presented")
    print()
    print(f"  deferral rate       {report.deferral_rate:>7.1%}   "
          f"{bar(report.deferral_rate)}")
    print(f"  miss rate           {report.miss_rate:>7.1%}   {bar(report.miss_rate)}")
    print(f"  detection accuracy  {report.detection_accuracy:>7.1%}   "
          f"{bar(report.detection_accuracy)}")
    print(f"    (the model naming the object -- reported for comparison only)")

    print("\n" + "-" * 78)
    print("  PER CLASS")
    print("-" * 78)
    print(f"  {'class':<16}{'n':>4}{'detected':>10}{'sorted ok':>11}"
          f"{'wrong bin':>11}{'deferred':>10}{'avg conf':>10}")
    for name, c in sorted(report.by_class.items()):
        acc = "  n/a" if c["sort_accuracy"] is None else f"{c['sort_accuracy']:.0%}"
        print(f"  {name:<16}{c['n']:>4}{c['object_correct']:>10}"
              f"{c['sorted_correct']:>11}{c['sorted_wrong']:>11}"
              f"{c['deferred']:>10}{c['avg_confidence']:>10.2f}")
        if c["n"] < args.min_per_class:
            print(f"    ^ only {c['n']} images; treat this row as indicative")

    print("\n" + "-" * 78)
    print("  WHERE ITEMS ACTUALLY WENT   (rows: expected bin)")
    print("-" * 78)
    for expected, landed in sorted(report.confusion.items()):
        parts = ", ".join(f"{k} {v}" for k, v in sorted(landed.items(),
                                                        key=lambda kv: -kv[1]))
        print(f"  {expected:<16} -> {parts}")

    if report.by_rule:
        print("\n" + "-" * 78)
        print("  WHICH RULE DECIDED")
        print("-" * 78)
        for rule, n in report.by_rule.most_common():
            print(f"  {rule:<28}{n:>5}   {bar(n / report.total, 18)}")

    print("\n" + "-" * 78)
    print("  DOES THE CONFIDENCE THRESHOLD EARN ITS PLACE?")
    print("-" * 78)
    print(f"  {'threshold':>10}{'n above':>9}{'accuracy':>10}{'coverage':>10}")
    for row in report.confidence_bands:
        acc = "   n/a" if row["accuracy"] is None else f"{row['accuracy']:>9.0%}"
        print(f"  {row['threshold']:>10.2f}{row['n_above']:>9}{acc}"
              f"{row['coverage']:>10.0%}")
    print("\n  Read the trade, not a single row: a higher threshold sorts less")
    print("  and sorts it better. The configured value is an operational")
    print("  choice about which error costs more, and this table is what")
    print("  makes that choice arguable rather than arbitrary.")
    print("=" * 78)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.disable(logging.INFO)

    root = Path(args.dataset) if args.dataset else DATASET
    if not root.is_absolute():
        root = ROOT / root

    dataset = load_dataset(root)
    if not dataset:
        print(f"\n  No evaluation set at {root}\n")
        print("  Build one -- it takes about ten minutes:\n")
        print("    python -m evaluation.capture --label bottle")
        print("    python -m evaluation.capture --label banana")
        print("    python -m evaluation.capture --label cell_phone")
        print("    ...20 or so frames each, varying angle and lighting\n")
        print("  The folder name is the label, so there is nothing to annotate.")
        return 1

    cfg = Config.load(args.config)
    try:
        harness = Harness(cfg)
    except ModelUnavailable as exc:
        print(f"  {exc}")
        return 2

    total = sum(len(v) for v in dataset.values())
    print(f"\n  evaluating {total} images across {len(dataset)} classes")
    print(f"  model: {cfg.get('detection.model_path')}")
    print(f"  labels: {cfg.get('detection.label_map')}\n")

    unmapped = []
    for folder in dataset:
        note = describe(folder, harness.labels)
        if note:
            print(f"    note: {note}")
            if resolve(folder, harness.labels) is None:
                unmapped.append(folder)
    if unmapped:
        print(f"\n    {len(unmapped)} folder(s) are not in the label map. That is a")
        print("    valid case -- an unknown object should reach manual inspection --")
        print("    but if you meant a known class, check the spelling against")
        print(f"    {cfg.get('detection.label_map')}.\n")

    outcomes = []
    for folder, images in dataset.items():
        label = harness.resolve(folder)
        for path in images:
            outcome = harness.run_image(path, label)
            outcomes.append(outcome)
            if not args.quiet:
                mark = {"SORTED_CORRECT": "ok  ", "SORTED_WRONG": "WRONG",
                        "DEFERRED": "manual", "MISSED": "miss", "HELD": "held"}
                print(f"    {mark.get(outcome.verdict, '?'):<7}"
                      f"{label:<14}{outcome.predicted_label:<14}"
                      f"{outcome.confidence:>5.0%}  -> {outcome.destination}")

    report = build_report(outcomes)
    print_report(report, args)

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(cfg.get("detection.model_path")),
        "label_map": str(cfg.get("detection.label_map")),
        "thresholds": {"auto_sort": cfg.get("confidence.auto_sort"),
                       "verify": cfg.get("confidence.verify")},
        "summary": report.summary(),
        "by_class": report.by_class,
        "confusion": report.confusion,
        "by_rule": dict(report.by_rule),
        "threshold_sweep": report.confidence_bands,
        "outcomes": [vars(o) | {"verdict": o.verdict} for o in outcomes],
    }
    out = RESULTS / "evaluation.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  written to {out}")
    print("  the reporting view picks this up automatically\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
