"""
Turning a folder name into a label the system knows.

A small file for a trap that would otherwise poison the results silently.
COCO class names contain spaces -- `cell phone`, `wine glass`, `hot dog` --
and a folder called `cell_phone` is the natural thing for anyone to create.
The label map has never heard of `cell_phone`, so the expected destination
would be derived as MANUAL_CHECK, and every phone the system correctly sorted
to E_WASTE would be scored as an error.

Nothing would crash. The report would simply be wrong, which is worse.

So the dataset boundary normalises, and the evaluation refuses to score a
folder it cannot resolve rather than guessing at it. The runtime label map is
left alone -- normalising there would mean the live system quietly accepting
labels no detector produces.
"""

from __future__ import annotations

from core.labels import LabelMap


def candidates(folder: str) -> list[str]:
    """The forms a folder name might be trying to spell."""
    raw = folder.strip().lower()
    forms = [raw, raw.replace("_", " "), raw.replace("-", " "),
             raw.replace(" ", "_")]
    seen, out = set(), []
    for f in forms:
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def resolve(folder: str, labels: LabelMap) -> str | None:
    """The label this folder means, or None if the map does not know it.

    None is a real answer, not a failure: an object the label map has never
    heard of genuinely should reach manual inspection, and proving that is a
    legitimate evaluation case. The caller decides whether to score it or
    warn about it -- this function will not invent a match.
    """
    known = set(labels.certain) | set(labels.ambiguous) | set(labels.overrides)
    for form in candidates(folder):
        if form in known:
            return form
    return None


def describe(folder: str, labels: LabelMap) -> str:
    resolved = resolve(folder, labels)
    if resolved is None:
        return f"'{folder}' is not in the label map -- scored as an unmapped object"
    if resolved != folder.strip().lower():
        return f"'{folder}' read as '{resolved}'"
    return ""
