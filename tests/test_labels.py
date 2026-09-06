"""
Stage 5 tests: the perception seam.

The claim being tested is that swapping the detection model is a config
change. So these tests load the *waste* label map -- describing a model that
does not exist yet -- run the real Material and Classification agents against
it, and check they behave correctly on classes COCO has never heard of.

If that works, the only thing standing between this system and a
purpose-trained detector is the weights file.

Run:  python -m tests.test_labels
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import tempfile

from agents.classification_agent import ClassificationAgent
from agents.material_agent import MaterialAgent
from core.config import Config
from core.event_bus import EventBus
from core.labels import LabelMap, load_label_map
from core.messages import Category, Detection, Material
from core.routing import destination_for
from tests.test_pipeline import PASSED, FAILED, check

logging.disable(logging.CRITICAL)


def detect(label: str, confidence: float = 0.93) -> Detection:
    return Detection(1, label, confidence, (100, 100, 200, 200), 1)


def agents_for(label_map: str):
    cfg = Config.load()
    labels = load_label_map(cfg, label_map)
    bus = EventBus()
    return labels, MaterialAgent(bus, labels=labels), ClassificationAgent(bus, labels=labels)


def classify(material_agent, classifier, label: str, confidence: float = 0.93):
    det = detect(label, confidence)
    return classifier.classify(det, material_agent.identify(det))


def write_map(name: str, body: str) -> str:
    path = Path(tempfile.gettempdir()) / "waste_tests" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
#  The shipped maps
# ---------------------------------------------------------------------------

def test_coco_map_still_drives_the_agents():
    """Stage 1 behaviour must survive the move out of agent code."""
    _, material, classifier = agents_for("config/labels_coco.yaml")

    result = classify(material, classifier, "bottle", 0.96)
    check("an ambiguous bottle still generalises to RECYCLABLE",
          result.category is Category.RECYCLABLE, result.category.value)

    result = classify(material, classifier, "cup", 0.96)
    check("a cup still defers to a human",
          result.category is Category.MANUAL_CHECK, result.category.value)

    result = classify(material, classifier, "knife", 0.96)
    check("a knife is still hazardous by object override",
          result.category is Category.HAZARDOUS, result.category.value)

    result = classify(material, classifier, "banana", 0.96)
    check("a banana is still organic",
          result.category is Category.ORGANIC, result.category.value)


def test_unmapped_label_goes_to_a_human():
    _, material, classifier = agents_for("config/labels_coco.yaml")
    result = classify(material, classifier, "sofa", 0.99)
    check("a 99%-confident sofa is still sent to manual inspection",
          result.category is Category.MANUAL_CHECK, result.category.value)


# ---------------------------------------------------------------------------
#  The model that does not exist yet
# ---------------------------------------------------------------------------

def test_waste_map_handles_the_classes_coco_cannot_see():
    """The whole point of Stage 5, tested before the model exists."""
    _, material, classifier = agents_for("config/labels_waste.yaml")

    result = classify(material, classifier, "battery", 0.94)
    check("a battery is hazardous",
          result.category is Category.HAZARDOUS, result.category.value)
    check("and routes to the hazardous bin",
          destination_for(result.category).value == "HAZARDOUS")

    result = classify(material, classifier, "aluminium_can", 0.94)
    check("an aluminium can is metal",
          result.category is Category.METAL, result.category.value)

    result = classify(material, classifier, "cardboard", 0.94)
    check("cardboard is fibre",
          result.category is Category.PAPER, result.category.value)

    result = classify(material, classifier, "plastic_bag", 0.94)
    check("a plastic bag is recyclable",
          result.category is Category.RECYCLABLE, result.category.value)


def test_the_custom_model_collapses_the_bottle_ambiguity():
    """The single biggest gain over COCO."""
    _, material, classifier = agents_for("config/labels_waste.yaml")

    det = detect("pet_bottle", 0.95)
    result = material.identify(det)
    check("a purpose-trained model names PET outright",
          result.material is Material.PET_PLASTIC, result.material.value)
    check("with no ambiguity left to resolve", not result.candidates)

    det = detect("glass_bottle", 0.95)
    result = material.identify(det)
    check("and glass outright", result.material is Material.GLASS)


def test_object_overrides_survive_the_model_swap():
    _, material, classifier = agents_for("config/labels_waste.yaml")
    check("a syringe is hazardous whatever it is made of",
          classify(material, classifier, "syringe").category is Category.HAZARDOUS)
    check("a used tissue is residual, not fibre recovery",
          classify(material, classifier, "tissue").category is Category.REJECT)


# ---------------------------------------------------------------------------
#  Bad maps degrade safely
# ---------------------------------------------------------------------------

def test_missing_map_sends_everything_to_a_human():
    """The correct failure. A system that cannot interpret what it sees
    should ask a person, not guess."""
    _, material, classifier = agents_for("config/does_not_exist.yaml")
    result = classify(material, classifier, "banana", 0.99)
    check("with no label map, even a clear banana goes to manual check",
          result.category is Category.MANUAL_CHECK, result.category.value)


def test_unknown_material_name_is_dropped_not_guessed():
    path = write_map("bad_material", """
certain:
  widget: NOT_A_REAL_MATERIAL
  banana: ORGANIC
""")
    labels = load_label_map(Config.load(), path)
    check("an invalid material is dropped", "widget" not in labels.certain)
    check("and the valid entries beside it still load",
          labels.certain.get("banana") is Material.ORGANIC)


def test_single_candidate_ambiguity_is_rejected():
    """One candidate is not an ambiguity; accepting it would make the
    Classification Agent's convergence rule meaningless."""
    path = write_map("bad_ambiguity", """
ambiguous:
  thing:
    candidates: [GLASS]
    reason: "only one"
  proper:
    candidates: [GLASS, PET_PLASTIC]
    reason: "two"
""")
    labels = load_label_map(Config.load(), path)
    check("a one-candidate ambiguity is ignored", "thing" not in labels.ambiguous)
    check("a real one is kept", "proper" in labels.ambiguous)


def test_unknown_override_category_is_dropped():
    path = write_map("bad_override", """
object_overrides:
  thing: NOT_A_CATEGORY
  blade: HAZARDOUS
""")
    labels = load_label_map(Config.load(), path)
    check("an invalid override category is dropped", "thing" not in labels.overrides)
    check("and a valid one is kept",
          labels.overrides.get("blade") is Category.HAZARDOUS)


def test_agents_never_see_the_model_name():
    """The seam itself: no agent's *code* knows which detector produced a label.

    Docstrings are stripped before the check. Prose explaining that COCO's
    `bottle` class motivated the ambiguity handling is useful to a reader and
    is not a dependency; what would matter is executable code naming a model,
    a weights file, or the detection library.
    """
    import ast
    import inspect

    from agents import classification_agent, material_agent

    for module in (material_agent, classification_agent):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                continue
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)

        code = ast.unparse(tree).lower()
        leaks = [w for w in ("yolo", "coco", ".pt", "ultralytics") if w in code]
        check(f"{module.__name__.split('.')[-1]} code is model-agnostic",
              not leaks, f"found {leaks}" if leaks else "")


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
