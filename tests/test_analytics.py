"""
Stage 4 tests: routing, analytics and planning.

The interesting tests here are the negative ones. It is easy to write an
analytics agent that always has something to say; the work is in making it
shut up when the data does not support a claim, and in stopping the planning
agent from doing anything other than talking.

Run:  python -m tests.test_analytics
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import tempfile
from datetime import datetime, timedelta, timezone

from agents.analytics_agent import AnalyticsAgent
from agents.planning_agent import PlanningAgent
from agents.routing_agent import RoutingAgent
from core.config import Config
from core.event_bus import EventBus
from core.messages import (Action, ActionStatus, Category, Destination, Material,
                           Topic, WasteRecord)
from database.database import Database
from pipeline import Pipeline
from tests.test_pipeline import PASSED, FAILED, check

logging.disable(logging.CRITICAL)


def temp_db(name: str) -> tuple[Config, Database]:
    cfg = Config.load()
    path = Path(tempfile.gettempdir()) / "waste_tests" / f"a_{name}.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()
    cfg._data.setdefault("database", {})["path"] = str(path)
    db = Database(cfg)
    db.start_session("simulation", "test")
    return cfg, db


def add(db: Database, category: Category, count: int, *, confidence: float = 0.92,
        action: Action = Action.SORT, day_offset: int = 0,
        rule: str = "R7_APPROVED") -> None:
    """Write `count` finished items on a given day."""
    when = datetime.now(timezone.utc) - timedelta(days=day_offset)
    for i in range(count):
        stamp = (when + timedelta(seconds=i)).isoformat(timespec="milliseconds")
        record = WasteRecord(
            track_id=i, label="scripted", detection_confidence=confidence,
            material=Material.UNKNOWN, category=category, action=action,
            destination=Destination.RECYCLING, status=ActionStatus.OK,
            reason="test", safety_rule=rule, latency_ms=1.0, route="test route",
            timestamp=stamp,
        )
        db.insert_detection(record)


# ---------------------------------------------------------------------------
#  Agent 6 -- routing
# ---------------------------------------------------------------------------

def test_routing_table_loads():
    cfg = Config.load()
    agent = RoutingAgent(cfg, EventBus())
    check("a route exists for every waste category",
          len(agent.routes) == len(list(Category)),
          f"{len(agent.routes)} of {len(list(Category))}")
    route = agent.route_for(Category.E_WASTE)
    check("the chain describes the kind of destination",
          route is not None and "recycler" in route.chain.lower(),
          route.chain if route else "missing")


def test_routing_names_no_facility():
    """The rule that matters: describe the kind, never invent the name."""
    agent = RoutingAgent(Config.load(), EventBus())
    named = {c.value: r.facility for c, r in agent.routes.items() if r.facility}
    check("no facility is named anywhere in the shipped routing table",
          not named, f"named={named}")
    check("and the agent reports zero configured facilities",
          agent.configured_facilities == 0)


def test_routing_distinguishes_streams_sharing_a_bin():
    agent = RoutingAgent(Config.load(), EventBus())
    glass = agent.route_for(Category.GLASS)
    paper = agent.route_for(Category.PAPER)
    check("glass and paper share a bin but not a downstream route",
          glass.chain != paper.chain,
          f"glass={glass.chain[:34]}... paper={paper.chain[:34]}...")


def test_effective_stream_follows_a_safety_override():
    """A held item is not entering the composting chain."""
    check("an override to manual inspection changes the route",
          Pipeline._effective_stream(Category.ORGANIC, Destination.MANUAL_CHECK)
          is Category.MANUAL_CHECK)
    check("an override to hazardous changes the route",
          Pipeline._effective_stream(Category.RECYCLABLE, Destination.HAZARDOUS)
          is Category.HAZARDOUS)
    check("otherwise the fine-grained classification is kept",
          Pipeline._effective_stream(Category.GLASS, Destination.RECYCLING)
          is Category.GLASS)


# ---------------------------------------------------------------------------
#  Agent 9 -- analytics
# ---------------------------------------------------------------------------

def test_refuses_to_report_on_thin_data():
    cfg, db = temp_db("thin")
    add(db, Category.ORGANIC, 5)
    insights = AnalyticsAgent(cfg, EventBus(), db).analyse()
    check("thin data yields exactly one finding",
          len(insights) == 1, f"got {len(insights)}")
    check("and that finding is that there is not enough data",
          insights[0].kind == "COVERAGE" and insights[0].strength == "provisional",
          f"{insights[0].kind}/{insights[0].strength}")
    check("no composition claim is made from 5 items",
          not any(i.kind == "COMPOSITION" for i in insights))
    db.close()


def test_dominant_stream_carries_its_evidence():
    cfg, db = temp_db("dominant")
    add(db, Category.ORGANIC, 40)
    add(db, Category.RECYCLABLE, 10)
    insights = AnalyticsAgent(cfg, EventBus(), db).analyse()
    composition = [i for i in insights if i.kind == "COMPOSITION"]
    top = composition[0]
    check("the dominant stream is named", "Organic" in top.headline, top.headline)
    check("with the count, total and share attached",
          top.evidence.get("count") == 40 and top.evidence.get("total") == 50
          and abs(top.evidence.get("share") - 0.8) < 0.01,
          str(top.evidence))


def test_reports_the_negative_result():
    """An even spread is a finding, not a reason to crown the biggest bar."""
    cfg, db = temp_db("even")
    for category in (Category.ORGANIC, Category.RECYCLABLE, Category.PAPER,
                     Category.E_WASTE):
        add(db, category, 12)
    insights = AnalyticsAgent(cfg, EventBus(), db).analyse()
    headlines = [i.headline for i in insights]
    check("an even spread is reported as no stream dominating",
          any("No single waste stream dominates" in h for h in headlines),
          str(headlines))
    db.close()


def test_trend_needs_two_adequate_periods():
    cfg, db = temp_db("trend_thin")
    add(db, Category.ORGANIC, 30, day_offset=0)
    add(db, Category.ORGANIC, 4, day_offset=1)       # yesterday too thin
    insights = AnalyticsAgent(cfg, EventBus(), db).analyse()
    trend = [i for i in insights if i.kind == "TREND"]
    check("a thin previous period yields no trend claim",
          trend and trend[0].strength == "provisional"
          and "Not enough" in trend[0].headline,
          trend[0].headline if trend else "no trend insight")
    db.close()


def test_trend_reports_direction_when_supported():
    cfg, db = temp_db("trend_ok")
    add(db, Category.ORGANIC, 40, day_offset=0)
    add(db, Category.ORGANIC, 20, day_offset=1)
    insights = AnalyticsAgent(cfg, EventBus(), db).analyse()
    trend = [i for i in insights if i.kind == "TREND"][0]
    check("a supported comparison names the direction and the change",
          "risen" in trend.headline and trend.evidence["latest"] == 40
          and trend.evidence["previous"] == 20,
          f"{trend.headline} {trend.evidence}")
    db.close()


def test_never_claims_causation():
    """The sentence this agent is not allowed to write."""
    cfg, db = temp_db("cause")
    # Yesterday: confident and clean. Today: less confident, more manual.
    add(db, Category.ORGANIC, 30, confidence=0.95, day_offset=1)
    add(db, Category.ORGANIC, 18, confidence=0.70, day_offset=0)
    add(db, Category.MANUAL_CHECK, 18, confidence=0.55, day_offset=0,
        action=Action.MANUAL_CHECK, rule="R4_UNCERTAIN")

    insights = AnalyticsAgent(cfg, EventBus(), db).analyse()
    association = [i for i in insights
                   if i.kind == "QUALITY" and "confidence_now" in i.evidence]
    check("the association between the two is reported",
          bool(association), f"kinds={[i.kind for i in insights]}")
    if association:
        text = (association[0].headline + " " + association[0].detail).lower()
        check("but never as a cause",
              "because" not in text, association[0].detail[:70])
        check("and the alternative explanations are named",
              "lighting" in text and "camera" in text)
    db.close()


def test_capacity_pressure_is_reported():
    cfg, db = temp_db("capacity")
    add(db, Category.ORGANIC, 30)

    class FullBin:
        destination = type("D", (), {"value": "ORGANIC"})()
        capacity, current_level, status = 100, 90, "NEARING CAPACITY"

    class Bins:
        bins = {"ORGANIC": FullBin()}

    db.snapshot_bins(Bins())
    insights = AnalyticsAgent(cfg, EventBus(), db).analyse()
    capacity = [i for i in insights if i.kind == "CAPACITY"]
    check("a bin above the pressure threshold is reported",
          bool(capacity), f"kinds={[i.kind for i in insights]}")
    check("and capacity findings are ordered first",
          insights[0].kind == "CAPACITY", insights[0].kind)
    db.close()


# ---------------------------------------------------------------------------
#  Agent 10 -- planning
# ---------------------------------------------------------------------------

def test_planning_cannot_act():
    """The structural guarantee, not a policy promise."""
    cfg, db = temp_db("planning")
    add(db, Category.ORGANIC, 40)
    bus = EventBus()
    insights = AnalyticsAgent(cfg, bus, db).analyse(announce=False)

    seen: list[str] = []
    bus.subscribe("*", lambda e: seen.append(e.event))
    planner = PlanningAgent(cfg, bus, RoutingAgent(cfg, bus))
    planner.recommend(insights)

    action_topics = {Topic.SORT_EXECUTED, Topic.SORT_DECISION, Topic.SAFETY_VERDICT}
    check("the planner holds no controller",
          not any("controller" in a or "actor" in a for a in vars(planner)),
          str(sorted(vars(planner))))
    check("and publishes nothing that could cause an action",
          not (set(seen) & action_topics), f"published={sorted(set(seen))}")
    check("only recommendations",
          set(seen) <= {Topic.RECOMMENDATION}, f"published={sorted(set(seen))}")
    db.close()


def test_every_recommendation_names_an_owner():
    cfg, db = temp_db("owners")
    add(db, Category.ORGANIC, 40)
    bus = EventBus()
    insights = AnalyticsAgent(cfg, bus, db).analyse(announce=False)
    recs = PlanningAgent(cfg, bus, RoutingAgent(cfg, bus)).recommend(insights)

    check("recommendations were produced", bool(recs), f"{len(recs)}")
    check("every one names a human owner",
          all(r.owner and r.owner.strip() for r in recs),
          str([r.owner for r in recs]))
    check("and none of them is this system",
          not any("agent" in r.owner.lower() or "system" in r.owner.lower()
                  for r in recs),
          str([r.owner for r in recs]))
    db.close()


def test_thin_data_recommends_collecting_more():
    cfg, db = temp_db("thin_rec")
    add(db, Category.ORGANIC, 4)
    bus = EventBus()
    insights = AnalyticsAgent(cfg, bus, db).analyse(announce=False)
    recs = PlanningAgent(cfg, bus, RoutingAgent(cfg, bus)).recommend(insights)
    check("the only advice on thin data is to gather more",
          len(recs) == 1 and "longer" in recs[0].headline.lower(),
          recs[0].headline if recs else "none")
    db.close()


def test_recommends_filling_in_the_routing_table():
    cfg, db = temp_db("routing_gap")
    add(db, Category.ORGANIC, 40)
    bus = EventBus()
    insights = AnalyticsAgent(cfg, bus, db).analyse(announce=False)
    recs = PlanningAgent(cfg, bus, RoutingAgent(cfg, bus)).recommend(insights)
    check("an empty routing table is surfaced as a gap to fill",
          any("routing table" in r.headline.lower() for r in recs),
          str([r.headline for r in recs]))
    db.close()


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
