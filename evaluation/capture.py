"""
Build the evaluation set without it becoming a chore.

    python -m evaluation.capture --label bottle

Opens the camera, draws what the detector currently sees, and saves a frame
each time you press SPACE. Hold the object, move it, change the angle and the
lighting, press space twenty times, press Q. Repeat per class.

**Why it draws the detections while you capture.** You find out immediately
whether the model can see the object at all. If you hold up a battery and no
box appears, that is the Stage 5 limitation in front of you -- and it is far
better to learn it during capture than after labelling a hundred images.
A frame with no detection is still worth saving: a miss is a real outcome
and the evaluation counts it.

The folder name IS the label, so there is nothing to annotate afterwards:

    evaluation/dataset/
      bottle/001.jpg  002.jpg ...
      banana/001.jpg ...

That works because this system's output is a decision per object, not a box.
One object per frame, and the folder says what it should have been.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from core.config import Config
from core.labels import load_label_map
from core.routing import destination_for
from evaluation.labels import resolve
from vision.camera import Camera, CameraUnavailable
from vision.detector import Detector, ModelUnavailable

DATASET = ROOT / "evaluation" / "dataset"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Capture labelled evaluation frames")
    p.add_argument("--label", required=True,
                   help="the object class these frames contain, e.g. bottle")
    p.add_argument("--config", default=None)
    p.add_argument("--source", default=None, help="override camera.source")
    p.add_argument("--list", action="store_true",
                   help="list what the dataset holds and exit")
    return p.parse_args(argv)


def inventory() -> dict[str, int]:
    if not DATASET.exists():
        return {}
    return {d.name: len(list(d.glob("*.jpg")))
            for d in sorted(DATASET.iterdir()) if d.is_dir()}


def print_inventory() -> None:
    counts = inventory()
    if not counts:
        print(f"  no dataset yet at {DATASET}")
        return
    print(f"  {DATASET}")
    total = 0
    for name, n in counts.items():
        flag = "" if n >= 20 else "   <- thin, aim for 20+"
        print(f"    {name:<18} {n:>4} frames{flag}")
        total += n
    print(f"    {'TOTAL':<18} {total:>4}")


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list:
        print_inventory()
        return 0

    cfg = Config.load(args.config)
    if args.source is not None:
        cfg._data.setdefault("camera", {})["source"] = args.source

    labels = load_label_map(cfg)
    raw = args.label.strip().lower()
    label = resolve(raw, labels) or raw
    if label != raw:
        print(f"\n  '{raw}' read as '{label}' -- saving under that name so the")
        print("  evaluation resolves it to the right expected bin.")
    known = set(labels.certain) | set(labels.ambiguous) | set(labels.overrides)
    if label not in known:
        print(f"  '{label}' is not in {labels.source.name if labels.source else 'the label map'}.")
        print(f"  Known labels: {', '.join(sorted(known))}")
        print(f"\n  Capturing it anyway is fine -- an unmapped object SHOULD go to")
        print(f"  manual check, and proving that is a legitimate evaluation case.")

    out_dir = DATASET / label
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.jpg")))

    try:
        camera = Camera(cfg)
        detector = Detector(cfg)
    except (CameraUnavailable, ModelUnavailable, FileNotFoundError) as exc:
        print(f"  {exc}")
        return 2

    expected_bin = "?"
    material = labels.material_for(label)
    if material is not None:
        from agents.classification_agent import MATERIAL_TO_CATEGORY
        cat = MATERIAL_TO_CATEGORY.get(material)
        if cat:
            expected_bin = destination_for(cat).value

    print(f"\n  capturing '{label}'  ({existing} already)  ->  {out_dir}")
    print(f"  expected bin: {expected_bin}")
    print("  SPACE save    Q quit    vary the angle, distance and lighting\n")

    window = f"capture: {label}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    saved = 0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                break

            detections = detector.detect(frame, camera.frame_id)
            view = frame.copy()

            for det in detections:
                x1, y1, x2, y2 = det.bbox
                hit = det.label.lower() == label
                colour = (110, 190, 90) if hit else (150, 150, 150)
                cv2.rectangle(view, (x1, y1), (x2, y2), colour, 2 if hit else 1)
                cv2.putText(view, f"{det.label} {det.confidence:.0%}",
                            (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, colour, 1, cv2.LINE_AA)

            banner = f"{label}   saved {saved}   total {existing + saved}"
            if not detections:
                banner += "   [nothing detected -- still worth saving]"
            cv2.rectangle(view, (0, 0), (view.shape[1], 30), (28, 26, 24), -1)
            cv2.putText(view, banner, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (232, 232, 230), 1, cv2.LINE_AA)

            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" "):
                saved += 1
                path = out_dir / f"{existing + saved:03d}.jpg"
                # The RAW frame, never the annotated one -- the evaluation
                # must re-detect from scratch, not read our own boxes back.
                cv2.imwrite(str(path), frame)
                print(f"    saved {path.name}"
                      + (f"   ({detections[0].label} {detections[0].confidence:.0%})"
                         if detections else "   (no detection)"))
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\n  {saved} new frames for '{label}'\n")
    print_inventory()
    print("\n  when you have 20+ per class:  python -m evaluation.evaluate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
