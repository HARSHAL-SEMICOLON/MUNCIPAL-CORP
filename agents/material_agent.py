"""
AGENT 2 -- MATERIAL IDENTIFICATION AGENT

Answers "what is it made of?", and is allowed to answer "I cannot tell".

This is the agent that most justifies keeping object, material and category
apart. A COCO-trained detector reports the object class "bottle" and stops
there -- it has no view on whether that bottle is PET or glass, because both
were labelled `bottle` in the training data. Guessing would be worse than
useless: it would put glass in the plastics stream and look confident doing
it.

So an ambiguous label returns UNKNOWN together with the candidates it could
not separate. The Classification Agent can still make progress when every
candidate leads to the same waste stream, and defers to a human when they do
not. Nothing in this system picks a material at random.

The vocabulary itself lives in `config/labels_*.yaml`, not here. This agent
does not know which model produced the label it is given, which is what lets
a purpose-trained waste model replace COCO without any change to this file.
"""

from __future__ import annotations

from core.config import Config
from core.event_bus import EventBus
from core.labels import LabelMap, load_label_map
from core.messages import Detection, Material, MaterialResult, Topic

from agents.base import Agent


class MaterialAgent(Agent):
    name = "MaterialAgent"

    def __init__(self, bus: EventBus, cfg: Config | None = None,
                 labels: LabelMap | None = None):
        super().__init__(bus)
        # A caller may inject a map directly (the tests do); otherwise it is
        # loaded from whichever file config points at.
        if labels is not None:
            self.labels = labels
        elif cfg is not None:
            self.labels = load_label_map(cfg)
        else:
            self.labels = LabelMap()

    def identify(self, detection: Detection) -> MaterialResult:
        label = detection.label.lower()

        material = self.labels.material_for(label)
        if material is not None:
            result = MaterialResult(
                material=material,
                confidence=detection.confidence,
                reason=f"object class '{label}' maps to a single material",
            )
        else:
            ambiguity = self.labels.ambiguity_for(label)
            if ambiguity is not None:
                result = MaterialResult(
                    material=Material.UNKNOWN,
                    confidence=detection.confidence,
                    reason=ambiguity.reason,
                    candidates=ambiguity.candidates,
                )
            else:
                result = MaterialResult(
                    material=Material.UNKNOWN,
                    confidence=detection.confidence,
                    reason=f"no material rule for object class '{label}'",
                )

        self.emit(Topic.MATERIAL_IDENTIFIED, {
            "track_id": detection.track_id,
            "label": detection.label,
            "material": result.material,
            "candidates": [c.value for c in result.candidates],
            "reason": result.reason,
        })
        return result
