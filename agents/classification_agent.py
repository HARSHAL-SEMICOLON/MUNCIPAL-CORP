"""
AGENT 3 -- WASTE CLASSIFICATION AGENT

Maps object + material onto a waste stream.

The interesting rule is the one for ambiguous materials, and it is worth
reading closely because it is where this stops being a lookup table.

When the Material Agent cannot separate PET from glass, this agent does not
guess. It asks a different question: does the ambiguity actually change what
happens to the item? A PET bottle and a glass bottle are different materials
in different sub-streams, but both are dry recyclables that end up in the
same bin on their way to material recovery -- so the item can proceed, and
the honest thing to report is the parent stream, RECYCLABLE.

A cup is the opposite case. Plastic and paper cups are recyclable; a ceramic
one is not, and ceramic in the glass stream is a well-known contaminant. The
candidates lead to different bins, so the item goes to a human.

Same uncertainty, two different answers, because the consequence differs.
"""

from __future__ import annotations

from core.config import Config
from core.event_bus import EventBus
from core.labels import LabelMap, load_label_map
from core.messages import Category, Classification, Detection, Material, MaterialResult, Topic
from core.routing import destination_for, generalise

from agents.base import Agent

MATERIAL_TO_CATEGORY: dict[Material, Category] = {
    Material.PET_PLASTIC:   Category.RECYCLABLE,
    Material.MIXED_PLASTIC: Category.RECYCLABLE,
    Material.ALUMINIUM:     Category.METAL,
    Material.FERROUS_METAL: Category.METAL,
    Material.GLASS:         Category.GLASS,
    Material.CERAMIC:       Category.REJECT,
    Material.PAPER:         Category.PAPER,
    Material.CARDBOARD:     Category.PAPER,
    Material.ORGANIC:       Category.ORGANIC,
    Material.ELECTRONIC:    Category.E_WASTE,
    Material.CHEMICAL:      Category.HAZARDOUS,
    Material.TEXTILE:       Category.REJECT,
}

class ClassificationAgent(Agent):
    name = "ClassificationAgent"

    def __init__(self, bus: EventBus, cfg: Config | None = None,
                 labels: LabelMap | None = None):
        super().__init__(bus)
        # Object-level overrides -- sharps and the like -- come from the same
        # label map as the materials, so a new detector brings its own.
        if labels is not None:
            self.labels = labels
        elif cfg is not None:
            self.labels = load_label_map(cfg)
        else:
            self.labels = LabelMap()

    def classify(self, detection: Detection, material: MaterialResult) -> Classification:
        label = detection.label.lower()

        override = self.labels.override_for(label)
        if override is not None:
            result = Classification(
                override, detection.confidence,
                f"'{label}' is handled as {override.value} by rule, "
                f"whatever it is made of",
            )
        elif material.material is not Material.UNKNOWN:
            category = MATERIAL_TO_CATEGORY.get(material.material, Category.MANUAL_CHECK)
            result = Classification(
                category, detection.confidence,
                f"{material.material.value} belongs to the {category.value} stream",
            )
        elif material.candidates:
            result = self._resolve_ambiguity(detection, material)
        else:
            result = Classification(
                Category.MANUAL_CHECK, detection.confidence,
                f"unknown material for '{label}'; no stream can be assigned",
            )

        self.emit(Topic.WASTE_CLASSIFIED, {
            "track_id": detection.track_id,
            "label": detection.label,
            "material": material.material,
            "category": result.category,
            "reason": result.reason,
        })
        return result

    def _resolve_ambiguity(self, detection: Detection,
                           material: MaterialResult) -> Classification:
        categories = {MATERIAL_TO_CATEGORY.get(c, Category.MANUAL_CHECK)
                      for c in material.candidates}
        bins = {destination_for(c) for c in categories}
        names = " or ".join(c.value for c in material.candidates)

        if len(bins) == 1:
            # Same bin either way, so the ambiguity is real but harmless.
            category = generalise(categories) or next(iter(categories))
            return Classification(
                category, detection.confidence,
                f"material is {names}; every possibility is handled as "
                f"{category.value}, so the ambiguity does not change the outcome",
            )

        streams = " / ".join(sorted(b.value for b in bins))
        return Classification(
            Category.MANUAL_CHECK, detection.confidence,
            f"material is {names}, which would route to {streams}; "
            f"the ambiguity changes the outcome, so a human decides",
        )
