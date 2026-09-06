"""
AGENT 10 -- MUNICIPAL PLANNING AGENT

Turns findings into suggestions for a person.

**This agent cannot act, by construction rather than by policy.** It is
handed insights and a routing table. It has no controller, no conveyor, no
bus write to any action topic, and no method that changes anything. If it
decides the organic bin needs emptying more often, the only thing that
happens is that a sentence appears in a report addressed to whoever is
responsible for collection rounds.

That is a deliberate limit, not a missing feature. Adjusting a collection
schedule spends public money and moves people's shifts; it is not a decision
a camera on a conveyor belt gets to make because a number crossed a
threshold. So every recommendation names an owner, and the owner is never
this system.

Recommendations rest on insights, which rest on rows in the database. When
the Analytics Agent says it does not have enough data, this agent has exactly
one thing to suggest -- collect more of it.
"""

from __future__ import annotations

from core.config import Config
from core.event_bus import EventBus
from core.messages import Insight, Recommendation, Topic

from agents.base import Agent

PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

# Generic role descriptions, not named posts at any particular authority.
OPERATOR = "Line operator"
SUPERVISOR = "Plant supervisor"
PLANNER = "Municipal waste planning"


class PlanningAgent(Agent):
    name = "PlanningAgent"

    def __init__(self, cfg: Config, bus: EventBus, routing=None):
        super().__init__(bus)
        self.routing = routing
        self.max_recommendations = int(cfg.get("planning.max_recommendations", 6))
        self.capacity_pressure = float(cfg.get("planning.capacity_pressure", 0.75))

    def recommend(self, insights: list[Insight],
                  announce: bool = True) -> list[Recommendation]:
        out: list[Recommendation] = []

        for insight in insights:
            out.extend(self._from_insight(insight))

        gap = self._routing_gap(insights)
        if gap is not None:
            out.append(gap)

        out.sort(key=lambda r: PRIORITY_ORDER.get(r.priority, 9))
        out = out[:self.max_recommendations]

        if announce:
            for rec in out:
                self.emit(Topic.RECOMMENDATION, {
                    "priority": rec.priority,
                    "owner": rec.owner,
                    "reason": rec.headline,
                    "consider": rec.consider,
                    "basis": list(rec.basis),
                })
        return out

    # -- mapping -----------------------------------------------------------

    def _from_insight(self, insight: Insight) -> list[Recommendation]:
        kind = insight.kind
        evidence = insight.evidence

        if kind == "COVERAGE":
            return [Recommendation(
                headline="Run the line longer before drawing conclusions",
                rationale=insight.detail,
                consider=(f"Process at least {evidence.get('threshold', 20)} items "
                          f"in one session, or seed the record with "
                          f"`python -m tests.seed_demo_data`, before treating "
                          f"any figure here as representative."),
                owner=OPERATOR, basis=(insight.headline,), priority="MEDIUM",
            )]

        if kind == "CAPACITY":
            bins = ", ".join(evidence.keys())
            full = any(d.get("fraction", 0) >= 0.98 for d in evidence.values()
                       if isinstance(d, dict))
            return [Recommendation(
                headline=f"Review collection frequency for {bins}",
                rationale=insight.detail,
                consider=("Emptying this bin more often, or increasing its "
                          "capacity, if the pattern holds across several "
                          "collection periods. One shift is not a pattern."),
                owner=SUPERVISOR, basis=(insight.headline,),
                priority="HIGH" if full else "MEDIUM",
            )]

        if kind == "QUALITY":
            if "confidence_now" in evidence:
                return [Recommendation(
                    headline="Check the camera before changing anything else",
                    rationale=insight.detail,
                    consider=("Camera position and focus, lighting on the belt, "
                              "and whether objects are arriving overlapped. "
                              "These are the cheap explanations and should be "
                              "ruled out before the model is blamed or "
                              "thresholds are moved."),
                    owner=OPERATOR, basis=(insight.headline,), priority="HIGH",
                )]
            return [Recommendation(
                headline="Manual inspection is absorbing a large share of items",
                rationale=insight.detail,
                consider=(f"Whether the inspection station is staffed for "
                          f"{evidence.get('manual', 0)} items per session, and "
                          f"which rung is sending them there -- "
                          f"{evidence.get('leading_rule', 'unknown')} is the "
                          f"most frequent. A rung firing constantly is either "
                          f"a real problem upstream or a threshold set wrong."),
                owner=SUPERVISOR, basis=(insight.headline,), priority="MEDIUM",
            )]

        if kind == "TREND" and insight.strength == "supported":
            change = evidence.get("change", 0)
            if change >= 0:
                return [Recommendation(
                    headline="Check downstream capacity is keeping pace",
                    rationale=insight.detail,
                    consider=("Whether collection and processing downstream can "
                              "absorb the higher volume, before the bins on the "
                              "line become the constraint."),
                    owner=PLANNER, basis=(insight.headline,), priority="MEDIUM",
                )]
            return []

        if kind == "COMPOSITION":
            return self._composition(insight)

        return []

    def _composition(self, insight: Insight) -> list[Recommendation]:
        evidence = insight.evidence
        stream = evidence.get("stream")

        if stream == "ORGANIC":
            return [Recommendation(
                headline="Review wet-waste processing capacity",
                rationale=insight.detail,
                consider=("Whether composting or biomethanation capacity matches "
                          "the organic share, if it stays at this level across "
                          "several collection periods."),
                owner=PLANNER, basis=(insight.headline,), priority="LOW",
            )]

        if stream in {"RECYCLABLE", "GLASS", "METAL", "PAPER"}:
            return [Recommendation(
                headline="Review dry-waste collection and recovery capacity",
                rationale=insight.detail,
                consider=("Whether material recovery downstream is sized for "
                          "this share of dry recyclables."),
                owner=PLANNER, basis=(insight.headline,), priority="LOW",
            )]

        if stream == "REJECT":
            return [Recommendation(
                headline="Investigate why residual waste is the largest stream",
                rationale=insight.detail,
                consider=("Residual is the stream to minimise, so a dominant "
                          "share is worth investigating rather than "
                          "accommodating. Check whether items are genuinely "
                          "non-recoverable or are being classified as reject "
                          "for want of a better rule."),
                owner=PLANNER, basis=(insight.headline,), priority="MEDIUM",
            )]

        # Dry-share insight, which reports a share rather than a single stream.
        if "dry" in evidence and evidence.get("share", 0) >= 0.4:
            return [Recommendation(
                headline="Material recovery is worth prioritising",
                rationale=insight.detail,
                consider=("Recoverable material is a large share of what the "
                          "line handles; the return on improving separation "
                          "quality is correspondingly larger."),
                owner=PLANNER, basis=(insight.headline,), priority="LOW",
            )]
        return []

    def _routing_gap(self, insights: list[Insight]) -> Recommendation | None:
        """A stream carrying real volume with no downstream route recorded.

        Not a criticism of the routing file -- it ships empty on purpose. But
        a report that cannot say where the hazardous stream goes is missing
        the part a municipality actually needs.
        """
        if self.routing is None or not self.routing.routes:
            return None
        if self.routing.configured_facilities:
            return None

        counts: dict[str, int] = {}
        for insight in insights:
            if insight.kind == "COMPOSITION":
                streams = insight.evidence.get("streams")
                if isinstance(streams, dict):
                    counts.update(streams)
                stream = insight.evidence.get("stream")
                if stream:
                    counts[stream] = insight.evidence.get("count", 0)
        if not counts:
            return None

        return Recommendation(
            headline="Add your municipality's downstream facilities to the routing table",
            rationale=("config/routing.yaml describes the kind of destination "
                       "each stream goes to, but every facility field is empty. "
                       "It ships that way deliberately: naming a recycler that "
                       "has not agreed to accept a stream would be inventing a "
                       "fact about a third party."),
            consider=("Filling in the facility, contact and licence fields with "
                      "verified local data. The routing agent will then report "
                      "exactly what you enter, and reports become actionable "
                      "rather than generic."),
            owner=PLANNER, priority="LOW",
            basis=tuple(i.headline for i in insights if i.kind == "COMPOSITION")[:1],
        )
