"""
Train a custom waste detector.

    python -m training.train --data training/dataset.yaml --epochs 100

This wraps Ultralytics training so the result drops straight into the system:
the class names in your dataset YAML must match the keys in
`config/labels_waste.yaml`, and when training finishes this script tells you
the two config lines to change. Nothing above the detector is touched.

**Read this before starting a run.** Training is the one part of this project
that hardware actually gates, and the script checks before it begins:

  * On a GPU, a yolov8n run over a few thousand images is roughly an hour.
  * On CPU it is an overnight job at best. This machine is CPU-only, so the
    script asks for --i-know-its-cpu before it will start one, rather than
    quietly occupying the laptop for eight hours.

The realistic route for a college project is a free Colab or Kaggle GPU
session: run this same script there, download `best.pt`, and drop it in
`models/`. The training is portable; the architecture it plugs into is
already finished and tested.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from core.config import Config
from core.labels import load_label_map


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a custom waste detector")
    p.add_argument("--data", default="training/dataset.yaml",
                   help="YOLO dataset YAML (paths + class names)")
    p.add_argument("--model", default="yolov8n.pt",
                   help="starting weights; yolov8n is the right size for a "
                        "laptop and for a Phase 2 microcontroller-driven line")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="waste-v1")
    p.add_argument("--device", default=None,
                   help="cuda device index, or 'cpu'. Default: auto")
    p.add_argument("--i-know-its-cpu", action="store_true",
                   help="proceed with CPU training despite the time cost")
    p.add_argument("--label-map", default="config/labels_waste.yaml",
                   help="the label map the TRAINED model will use -- checked "
                        "against the dataset class names. Defaults to the "
                        "waste map, not whichever map config.yaml has active.")
    p.add_argument("--check", action="store_true",
                   help="validate the dataset and class names, then stop")
    return p.parse_args(argv)


def describe_hardware() -> tuple[str, bool]:
    try:
        import torch
    except ImportError:
        return "torch is not installed", False
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA GPU: {name} ({memory:.1f} GB)", True
    import os
    return (f"CPU only ({os.cpu_count()} cores, torch {torch.__version__})",
            False)


def check_classes(data_path: Path, label_map: str) -> int:
    """Cross-check the dataset's class names against the label map.

    A class the label map has never heard of trains fine and then produces
    UNKNOWN for every instance at run time -- a silent failure that would only
    surface as an unexplained rise in manual inspections. Better to catch it
    before spending a night on the training.
    """
    if not data_path.exists():
        print(f"  dataset YAML not found: {data_path}")
        print(f"  copy training/dataset.yaml and fill in your paths")
        return 1

    with open(data_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    names = data.get("names") or {}
    if isinstance(names, dict):
        class_names = [str(v) for _, v in sorted(names.items())]
    else:
        class_names = [str(v) for v in names]

    if not class_names:
        print(f"  {data_path} lists no class names")
        return 1

    cfg = Config.load()
    labels = load_label_map(cfg, label_map)
    known = set(labels.certain) | set(labels.ambiguous) | set(labels.overrides)

    unknown = [n for n in class_names if n.lower() not in known]
    unused = sorted(known - {n.lower() for n in class_names})

    print(f"  dataset classes : {len(class_names)}")
    print(f"  label map       : {labels.source.name if labels.source else '?'} "
          f"({len(known)} labels)")

    if unknown:
        print(f"\n  {len(unknown)} dataset class(es) the label map does not know:")
        for name in unknown:
            print(f"    {name}")
        print("\n  Every instance of these would be reported UNKNOWN at run time")
        print("  and sent to manual inspection. Add them to the label map first.")
        return 1

    print("  every dataset class has a material rule")
    if unused:
        print(f"\n  {len(unused)} label-map entries have no class in this dataset")
        print(f"  (harmless -- they simply never fire): {', '.join(unused[:8])}"
              + (" ..." if len(unused) > 8 else ""))
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    print("=" * 70)
    print("  WASTE DETECTOR TRAINING")
    print("=" * 70)

    hardware, has_gpu = describe_hardware()
    print(f"\n  hardware        : {hardware}")

    print()
    problems = check_classes(data_path, args.label_map)
    if problems or args.check:
        return problems

    device = args.device or ("0" if has_gpu else "cpu")
    if device == "cpu" and not args.i_know_its_cpu:
        print("\n" + "-" * 70)
        print("  Refusing to start CPU training without --i-know-its-cpu.")
        print()
        print("  A yolov8n run of 100 epochs on a few thousand images takes")
        print("  roughly an hour on a modest GPU and most of a night on CPU.")
        print("  That is a real cost to pay by accident.")
        print()
        print("  Better options:")
        print("    * Run this same script on a free Colab or Kaggle GPU,")
        print("      then copy best.pt into models/")
        print("    * Or start it here deliberately, overnight:")
        print(f"      python -m training.train --data {args.data} "
              f"--i-know-its-cpu")
        print("-" * 70)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n  ultralytics is not installed. pip install ultralytics")
        return 3

    print(f"\n  starting        : {args.model}, {args.epochs} epochs, "
          f"imgsz {args.imgsz}, device {device}")
    print("=" * 70 + "\n")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        project=str(ROOT / "training" / "runs"),
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print("\n" + "=" * 70)
    if best.exists():
        target = ROOT / "models" / f"{args.name}.pt"
        shutil.copy(best, target)
        print(f"  best weights copied to {target}")
        print("\n  To use them, change two lines in config/config.yaml:")
        print(f"\n    detection:")
        print(f'      model_path: "models/{args.name}.pt"')
        print(f'      label_map:  "{args.label_map}"')
        print("\n  That is the whole integration. No agent code changes.")
    else:
        print(f"  training finished but no best.pt at {best}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
