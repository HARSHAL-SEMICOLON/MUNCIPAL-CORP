"""
The label map: what a detector's class names mean.

This is the seam between the model and the agents, and it is the second one
in the system worth naming. The first (hardware/controller.py) makes the
actuator replaceable; this one makes the *detector* replaceable.

Before Stage 5 the COCO class names were written into the Material and
Classification agents. That worked while there was one model, and would have
meant editing agent logic the day a purpose-trained waste model arrived --
the same mistake the hardware seam exists to prevent, one layer up.

So swapping models is now two lines of config:

    detection:
      model_path: "models/waste-v1.pt"
      label_map:  "config/labels_waste.yaml"

The agents never learn which model produced a label. A map that is missing,
malformed, or names a material the vocabulary does not know degrades to
"everything is UNKNOWN" -- which routes items to manual inspection. That is
the correct failure: a system that cannot interpret what it sees should ask
a person, not guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from core.config import Config
from core.messages import Category, Material

log = logging.getLogger("labels")


@dataclass(frozen=True)
class Ambiguity:
    candidates: tuple[Material, ...]
    reason: str


@dataclass
class LabelMap:
    """One detector's vocabulary, translated into materials and overrides."""
    certain: dict[str, Material] = field(default_factory=dict)
    ambiguous: dict[str, Ambiguity] = field(default_factory=dict)
    overrides: dict[str, Category] = field(default_factory=dict)
    source: Path | None = None

    @property
    def known_labels(self) -> int:
        return len(self.certain) + len(self.ambiguous)

    def material_for(self, label: str) -> Material | None:
        return self.certain.get(label.lower())

    def ambiguity_for(self, label: str) -> Ambiguity | None:
        return self.ambiguous.get(label.lower())

    def override_for(self, label: str) -> Category | None:
        return self.overrides.get(label.lower())


def load_label_map(cfg: Config, path: str | Path | None = None) -> LabelMap:
    """Load the active map, or an explicit one.

    `path` is for tools that need to inspect a map other than the one
    currently in use -- the trainer checks a dataset against the map the
    *trained* model will use, not the COCO map still in config.
    """
    if path is not None:
        path = Path(path)
        if not path.is_absolute():
            path = Path(cfg.source).resolve().parent.parent / path
    else:
        path = cfg.path("detection.label_map", "config/labels_coco.yaml")

    if not path.exists():
        log.error("No label map at %s; every object will be reported as "
                  "UNKNOWN and sent to manual inspection", path)
        return LabelMap(source=path)

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError:
        log.exception("Could not parse %s; every object will be UNKNOWN", path)
        return LabelMap(source=path)

    certain: dict[str, Material] = {}
    for label, name in (data.get("certain") or {}).items():
        material = _material(name, f"certain.{label}", path)
        if material is not None:
            certain[str(label).lower()] = material

    ambiguous: dict[str, Ambiguity] = {}
    for label, spec in (data.get("ambiguous") or {}).items():
        names = (spec or {}).get("candidates") or []
        candidates = tuple(m for m in
                           (_material(n, f"ambiguous.{label}", path) for n in names)
                           if m is not None)
        if len(candidates) < 2:
            # A single candidate is not an ambiguity; it is a certain mapping
            # written in the wrong place, and silently accepting it would make
            # the Classification Agent's convergence rule meaningless.
            log.warning("%s: ambiguous.%s needs at least two candidates; ignoring",
                        path.name, label)
            continue
        ambiguous[str(label).lower()] = Ambiguity(
            candidates=candidates,
            reason=str((spec or {}).get("reason", "")).strip()
                   or f"the detector class '{label}' covers several materials",
        )

    overrides: dict[str, Category] = {}
    for label, name in (data.get("object_overrides") or {}).items():
        try:
            overrides[str(label).lower()] = Category(str(name))
        except ValueError:
            log.warning("%s: object_overrides.%s names an unknown category %r; "
                        "ignoring", path.name, label, name)

    overlap = set(certain) & set(ambiguous)
    if overlap:
        log.warning("%s: %s appear in both certain and ambiguous; "
                    "certain wins", path.name, ", ".join(sorted(overlap)))
        for label in overlap:
            ambiguous.pop(label, None)

    log.info("Label map %s: %d certain, %d ambiguous, %d overrides",
             path.name, len(certain), len(ambiguous), len(overrides))
    return LabelMap(certain=certain, ambiguous=ambiguous, overrides=overrides,
                    source=path)


def _material(name, where: str, path: Path) -> Material | None:
    try:
        return Material(str(name))
    except ValueError:
        log.warning("%s: %s names an unknown material %r; ignoring. "
                    "Valid materials: %s", path.name, where, name,
                    ", ".join(m.value for m in Material))
        return None
