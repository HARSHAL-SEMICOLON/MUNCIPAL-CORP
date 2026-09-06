"""
Where a waste stream physically goes, and how streams relate to each other.

Kept out of the agents on purpose. Two agents need this map for different
reasons -- Classification asks whether an ambiguity actually matters, and
Decision asks which bin to name -- and a single source of truth is what stops
those two answers from drifting apart.

Stage 4's Municipal Routing Agent extends this file with the downstream chain
(material recovery facility, composting, authorised e-waste recycler). None of
that is invented here: the downstream table stays empty until real municipal
data is supplied.
"""

from __future__ import annotations

from core.messages import Category, Destination

# A category is a waste STREAM. A destination is a BIN. They are not the same:
# four different streams share the recycling bin because they all go on to
# material recovery, and keeping them distinct is what lets the system report
# "3 kg of glass" rather than only "3 kg of recyclables".
CATEGORY_TO_DESTINATION: dict[Category, Destination] = {
    Category.RECYCLABLE:   Destination.RECYCLING,
    Category.GLASS:        Destination.RECYCLING,
    Category.METAL:        Destination.RECYCLING,
    Category.PAPER:        Destination.RECYCLING,
    Category.ORGANIC:      Destination.ORGANIC,
    Category.E_WASTE:      Destination.E_WASTE,
    Category.HAZARDOUS:    Destination.HAZARDOUS,
    Category.REJECT:       Destination.REJECT,
    Category.MANUAL_CHECK: Destination.MANUAL_CHECK,
}

# Glass, metal and paper are refinements of the dry-recyclable stream. When
# the system cannot tell which refinement applies but knows they are all dry
# recyclables, naming the parent is accurate rather than evasive.
PARENT_CATEGORY: dict[Category, Category] = {
    Category.GLASS: Category.RECYCLABLE,
    Category.METAL: Category.RECYCLABLE,
    Category.PAPER: Category.RECYCLABLE,
}


def destination_for(category: Category) -> Destination:
    return CATEGORY_TO_DESTINATION.get(category, Destination.MANUAL_CHECK)


def generalise(categories: set[Category]) -> Category | None:
    """The one category covering all of these, or None if they truly differ.

    {RECYCLABLE, GLASS} -> RECYCLABLE   (both are dry recyclables)
    {RECYCLABLE, REJECT} -> None        (different bins, ask a human)
    """
    if not categories:
        return None
    roots = {PARENT_CATEGORY.get(c, c) for c in categories}
    return roots.pop() if len(roots) == 1 else None
