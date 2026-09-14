"""
Pick the demo's sample images, by asking the detector which ones it can see.

    python -m tools.make_samples

`dashboard/samples/` is what a first-time visitor clicks before they upload
anything of their own, so it is the first impression the project makes. Choosing
those frames by eye is a trap: a photograph that looks obvious to a person can be
one the model scores at 0.3, and then the demo opens on "nothing detected" and
looks broken. That exact failure is why this script exists.

So the frames are chosen by the detector. This reads the evaluation set you
captured with `evaluation.capture`, runs the real detector over every frame, and
copies the single highest-confidence frame per class into `dashboard/samples/`.
The sample set is then, by construction, the set the demo handles best.

It refuses to copy a class whose best frame is below the inspection threshold,
and says so, because shipping a sample the system cannot see is worse than
shipping one fewer sample.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from core.config import Config
from vision.detector import Detector, ModelUnavailable

DATASET = ROOT / "evaluation" / "dataset"
SAMPLES = ROOT / "dashboard" / "samples"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Choose demo samples by detector confidence")
    p.add_argument("--min-confidence", type=float, default=0.60,
                   help="skip a class whose best frame scores below this "
                        "(default matches confidence.verify: below it the "
                        "Safety Agent sends the item to a human anyway)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be chosen without writing anything")
    return p.parse_args(argv)


def best_frame(detector: Detector, folder: Path, wanted: str):
    """The frame in `folder` the detector is most confident about.

    Scored on the detection whose label matches the folder name, not simply the
    highest-scoring box in the frame -- a picture of a bottle where the model is
    95% sure about the table behind it is not a good bottle sample.
    """
    best = None
    for image_path in sorted(folder.glob("*.jpg")):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        for det in detector.detect(frame, frame_id=0):
            if det.label.lower() != wanted.lower():
                continue
            if best is None or det.confidence > best[1]:
                best = (image_path, det.confidence)
    return best


def main(argv=None) -> int:
    args = parse_args(argv)

    if not DATASET.exists():
        print(f"No evaluation set at {DATASET}.")
        print("Capture some frames first, e.g.:")
        print("    python -m evaluation.capture --label bottle")
        return 1

    try:
        detector = Detector(Config.load())
    except ModelUnavailable as exc:
        print(f"Detection model unavailable: {exc}")
        return 2

    classes = sorted(d for d in DATASET.iterdir() if d.is_dir())
    if not classes:
        print(f"{DATASET} has no class folders yet.")
        return 1

    SAMPLES.mkdir(parents=True, exist_ok=True)
    chosen, skipped = 0, 0

    for folder in classes:
        label = folder.name
        frames = list(folder.glob("*.jpg"))
        if not frames:
            print(f"  {label:16s} no frames")
            continue

        best = best_frame(detector, folder, label)
        if best is None:
            print(f"  {label:16s} SKIPPED -- the detector never reported "
                  f"'{label}' in {len(frames)} frames")
            skipped += 1
            continue

        path, confidence = best
        if confidence < args.min_confidence:
            print(f"  {label:16s} SKIPPED -- best frame only {confidence:.0%} "
                  f"(below {args.min_confidence:.0%})")
            skipped += 1
            continue

        target = SAMPLES / f"{label.replace(' ', '_')}.jpg"
        if not args.dry_run:
            shutil.copy(path, target)
        print(f"  {label:16s} {confidence:.0%}  <- {path.name}"
              f"{' (dry run)' if args.dry_run else ''}")
        chosen += 1

    print()
    print(f"{chosen} sample(s) chosen, {skipped} skipped.")
    if chosen and not args.dry_run:
        print(f"Written to {SAMPLES}.")
        print("These are committed, so check none of them show anything "
              "private before you push.")
    if skipped:
        print()
        print("A skipped class is the honest Stage 5 limitation showing: the "
              "current COCO model cannot see it. Train waste-v1.pt and run "
              "this again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
