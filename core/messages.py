"""
Structured messages exchanged between agents.

Every agent speaks only in these types. That is what makes the architecture
survivable: swapping the detector, the dashboard or the actuator changes the
code that PRODUCES these objects, never the code that consumes them.

Three ideas are kept deliberately separate throughout:

    OBJECT    what the camera saw          "bottle"
    MATERIAL  what it is made of           PET plastic / glass / UNKNOWN
    CATEGORY  which waste stream it enters RECYCLABLE

They are not the same thing, and collapsing them is the usual reason a
sorting system cannot explain itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ---------------------------------------------------------------------------
#  Vocabularies
# ---------------------------------------------------------------------------

class Material(str, Enum):
    PET_PLASTIC   = "PET_PLASTIC"
    MIXED_PLASTIC = "MIXED_PLASTIC"
    ALUMINIUM     = "ALUMINIUM"
    FERROUS_METAL = "FERROUS_METAL"
    GLASS         = "GLASS"
    CERAMIC       = "CERAMIC"
    PAPER         = "PAPER"
    CARDBOARD     = "CARDBOARD"
    ORGANIC       = "ORGANIC"
    TEXTILE       = "TEXTILE"
    ELECTRONIC    = "ELECTRONIC"
    CHEMICAL      = "CHEMICAL"
    UNKNOWN       = "UNKNOWN"


class Category(str, Enum):
    RECYCLABLE   = "RECYCLABLE"
    ORGANIC      = "ORGANIC"
    E_WASTE      = "E_WASTE"
    HAZARDOUS    = "HAZARDOUS"
    GLASS        = "GLASS"
    METAL        = "METAL"
    PAPER        = "PAPER"
    REJECT       = "REJECT"
    MANUAL_CHECK = "MANUAL_CHECK"


class Action(str, Enum):
    SORT           = "SORT"
    MANUAL_CHECK   = "MANUAL_CHECK"
    HOLD           = "HOLD"
    REJECT         = "REJECT"
    STOP_CONVEYOR  = "STOP_CONVEYOR"
    ALERT_OPERATOR = "ALERT_OPERATOR"


class Destination(str, Enum):
    RECYCLING    = "RECYCLING"
    ORGANIC      = "ORGANIC"
    E_WASTE      = "E_WASTE"
    HAZARDOUS    = "HAZARDOUS"
    REJECT       = "REJECT"
    MANUAL_CHECK = "MANUAL_CHECK"
    NONE         = "NONE"


class ActionStatus(str, Enum):
    """Outcome of a physical (or simulated) actuation.

    Present from the first commit even though the simulation cannot fail,
    so the retry/hold logic above the seam is written once and never has to
    be retrofitted when a real servo starts missing its target.
    """
    OK      = "OK"
    FAILED  = "FAILED"
    TIMEOUT = "TIMEOUT"
    REFUSED = "REFUSED"      # controller declined (e.g. bin full, e-stop)


class Severity(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    ALERT    = "ALERT"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
#  Agent payloads
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Vision Agent output -- one tracked object in one frame."""
    track_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]          # x1, y1, x2, y2
    frame_id: int
    timestamp: str = field(default_factory=_now)

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class MaterialResult:
    """Material Agent output.

    `candidates` matters. A COCO-trained detector reports "bottle" without
    telling us whether it is PET or glass, so the honest answer is UNKNOWN
    with both possibilities listed -- never a coin flip between them. The
    Classification Agent can still proceed when every candidate leads to the
    same waste stream.
    """
    material: Material
    confidence: float
    reason: str
    candidates: tuple[Material, ...] = ()


@dataclass
class Classification:
    category: Category
    confidence: float
    reason: str


@dataclass
class Decision:
    """Decision Agent output -- a PROPOSAL, not a command.

    Nothing in the system acts on this object directly. It goes to the
    Safety Agent, which owns the only path to the Action Agent.
    """
    action: Action
    destination: Destination
    confidence: float
    reason: str
    proposed_by: str = "DecisionAgent"


@dataclass
class SafetyVerdict:
    """Safety Agent output -- the gate.

    `approved` False means the proposed decision was overridden; `action`
    and `destination` then carry what happens instead. `rule` names the rung
    that fired, so every override is explainable after the fact.
    """
    approved: bool
    action: Action
    destination: Destination
    reason: str
    rule: str
    severity: Severity = Severity.INFO


@dataclass
class ActionResult:
    status: ActionStatus
    destination: Destination
    latency_ms: float
    attempts: int = 1
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is ActionStatus.OK


@dataclass
class Route:
    """Municipal Routing Agent output -- where the stream goes after the bin.

    `facility` is None unless a municipality has filled in verified local
    data. The system describes the *kind* of destination ("authorised e-waste
    recycler") and refuses to name one it cannot vouch for.
    """
    stream: str
    stages: tuple[str, ...]
    facility: str | None = None
    notes: str = ""

    @property
    def chain(self) -> str:
        return " -> ".join(self.stages)


@dataclass
class Insight:
    """Analytics Agent output -- one finding, with the numbers behind it.

    `evidence` is not decoration. An insight that cannot show its working is
    indistinguishable from an invented one, and this system is not allowed to
    invent statistics.
    """
    headline: str
    detail: str
    evidence: dict[str, Any]
    period: str
    kind: str                     # COMPOSITION | TREND | QUALITY | CAPACITY
    strength: str = "supported"   # supported | provisional


@dataclass
class Recommendation:
    """Planning Agent output -- advice, addressed to a person.

    `owner` exists so no recommendation is ever ownerless. This agent has no
    controller, no actuator and no write path to any action topic; it cannot
    do the thing it suggests, by construction rather than by policy.
    """
    headline: str
    rationale: str
    consider: str
    owner: str
    basis: tuple[str, ...] = ()
    priority: str = "MEDIUM"      # LOW | MEDIUM | HIGH


@dataclass
class WasteRecord:
    """The full audit trail for one object, start to finish.

    One record per physical item -- not one per frame. This is what the
    database stores and what the Analytics Agent reads in later stages.
    """
    track_id: int
    label: str
    detection_confidence: float
    material: Material
    category: Category
    action: Action
    destination: Destination
    status: ActionStatus
    reason: str
    safety_rule: str
    latency_ms: float
    route: str = ""               # the downstream municipal chain, if known
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d


@dataclass
class Event:
    """Anything worth putting on the bus or in the log."""
    event: str
    agent: str
    payload: dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.INFO
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "agent": self.agent,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "payload": _plain(self.payload),
        }


def _plain(value: Any) -> Any:
    """Make a payload JSON-safe without the caller having to think about it."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    return value


# Event names, so publishers and subscribers cannot drift apart on spelling.
class Topic:
    WASTE_DETECTED     = "WASTE_DETECTED"
    MATERIAL_IDENTIFIED = "MATERIAL_IDENTIFIED"
    WASTE_CLASSIFIED   = "WASTE_CLASSIFIED"
    SORT_DECISION      = "SORT_DECISION"
    SAFETY_VERDICT     = "SAFETY_VERDICT"
    SORT_EXECUTED      = "SORT_EXECUTED"
    # One completed item, start to finish. The Monitoring Agent subscribes to
    # this rather than being called by the pipeline, which is what lets
    # persistence and analytics be added without touching the sorting path.
    WASTE_RECORDED     = "WASTE_RECORDED"
    ROUTE_ASSIGNED     = "ROUTE_ASSIGNED"
    INSIGHT_GENERATED  = "INSIGHT_GENERATED"
    RECOMMENDATION     = "RECOMMENDATION"
    SYSTEM_UPDATE      = "SYSTEM_UPDATE"
    BIN_ALERT          = "BIN_ALERT"
    SYSTEM_FAULT       = "SYSTEM_FAULT"
