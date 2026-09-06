"""
AGENT 9 -- ANALYTICS AGENT

Reads the record and says what is in it.

Everything below is computed from rows in the database. Nothing is estimated,
extrapolated or filled in, and the guards matter more than the arithmetic:

**It refuses on thin data.** A "trend" drawn from four items is not a trend,
and a system that says "organic waste is rising" after a two-minute run has
taught its operator to ignore it. Below `analytics.min_items_for_insight` this
agent returns one honest finding -- that there is not enough on record yet --
and stops. Period comparisons additionally require both periods to clear
`min_items_per_period`, so a full day is never compared against a lunch break.

**It reports association, not cause.** The obvious observation to make is
"manual inspection went up because confidence went down". This agent will not
say *because*. It reports that the two moved together and names both numbers,
because a camera that was nudged, a change of lighting and a genuinely harder
mix of waste all produce that same pair of movements, and the log cannot tell
them apart. Distinguishing them is the operator's job, and it starts with
being told what actually happened rather than why something guessed it did.

**Every insight carries its evidence.** An insight that cannot show its
working is indistinguishable from an invented one.
"""

from __future__ import annotations

from core.config import Config
from core.event_bus import EventBus
from core.messages import Insight, Topic

from agents.base import Agent

DRY_RECYCLABLE = {"RECYCLABLE", "GLASS", "METAL", "PAPER"}

# Attention-worthy findings first. Composition is the headline in a report;
# a bin about to overflow is the headline on a shift.
KIND_ORDER = {"CAPACITY": 0, "QUALITY": 1, "TREND": 2, "COMPOSITION": 3, "COVERAGE": 4}


def _pct(value: float) -> str:
    return f"{value:.0%}"


class AnalyticsAgent(Agent):
    name = "AnalyticsAgent"

    def __init__(self, cfg: Config, bus: EventBus, db):
        super().__init__(bus)
        self.db = db
        self.min_items = int(cfg.get("analytics.min_items_for_insight", 20))
        self.min_period = int(cfg.get("analytics.min_items_per_period", 15))
        self.dominant_share = float(cfg.get("analytics.dominant_share", 0.35))
        self.trend_change = float(cfg.get("analytics.trend_change", 0.20))
        self.manual_concern = float(cfg.get("analytics.manual_rate_concern", 0.25))
        self.capacity_pressure = float(cfg.get("planning.capacity_pressure", 0.75))

    # -- entry point -------------------------------------------------------

    def analyse(self, announce: bool = True) -> list[Insight]:
        if self.db is None:
            return []

        overall = self.db.window_summary()
        if overall["processed"] < self.min_items:
            insights = [self._not_enough_data(overall)]
        else:
            insights = [i for i in (
                self._composition(overall),
                self._dry_share(overall),
                self._trend(),
                self._manual_rate(overall),
                self._confidence_association(),
                self._capacity(),
            ) if i is not None]

        insights.sort(key=lambda i: KIND_ORDER.get(i.kind, 9))

        if announce:
            for insight in insights:
                self.emit(Topic.INSIGHT_GENERATED, {
                    "kind": insight.kind,
                    "strength": insight.strength,
                    "period": insight.period,
                    "reason": insight.headline,
                    "detail": insight.detail,
                    "evidence": insight.evidence,
                })
        return insights

    # -- findings ----------------------------------------------------------

    def _not_enough_data(self, overall: dict) -> Insight:
        need = self.min_items - overall["processed"]
        return Insight(
            headline="Not enough data yet for a reliable reading",
            detail=(f"{overall['processed']} items are on record; this agent "
                    f"reports findings from {self.min_items}. Roughly {need} "
                    f"more items will be enough. Nothing is being inferred "
                    f"from the run so far."),
            evidence={"processed": overall["processed"],
                      "threshold": self.min_items},
            period="all time", kind="COVERAGE", strength="provisional",
        )

    def _composition(self, overall: dict) -> Insight | None:
        counts = self.db.category_counts()
        if not counts:
            return None
        total = sum(counts.values())
        name, count = max(counts.items(), key=lambda kv: kv[1])
        share = count / total

        if share >= self.dominant_share:
            return Insight(
                headline=f"{name.replace('_', ' ').title()} is the largest waste stream",
                detail=(f"{count} of {total} items ({_pct(share)}) were assigned "
                        f"to {name}."),
                evidence={"stream": name, "count": count, "total": total,
                          "share": round(share, 3)},
                period="all time", kind="COMPOSITION",
            )

        # The honest negative result. "No stream dominates" is a finding, and
        # reporting it stops the largest bar being read as a headline.
        return Insight(
            headline="No single waste stream dominates",
            detail=(f"The largest is {name} at {count} of {total} items "
                    f"({_pct(share)}), below the {_pct(self.dominant_share)} "
                    f"share this agent needs before calling a stream dominant."),
            evidence={"largest": name, "count": count, "total": total,
                      "share": round(share, 3),
                      "threshold": self.dominant_share},
            period="all time", kind="COMPOSITION",
        )

    def _dry_share(self, overall: dict) -> Insight | None:
        counts = self.db.category_counts()
        total = sum(counts.values())
        if not total:
            return None
        dry = sum(v for k, v in counts.items() if k in DRY_RECYCLABLE)
        if not dry:
            return None
        share = dry / total
        parts = ", ".join(f"{k.lower()} {v}" for k, v in sorted(
            ((k, v) for k, v in counts.items() if k in DRY_RECYCLABLE),
            key=lambda kv: -kv[1]))
        return Insight(
            headline=f"Dry recyclables are {_pct(share)} of everything sorted",
            detail=(f"{dry} of {total} items across four recoverable streams "
                    f"({parts}). These share one bin but separate again "
                    f"downstream, which is why the category is recorded "
                    f"alongside the destination."),
            evidence={"dry": dry, "total": total, "share": round(share, 3),
                      "streams": {k: v for k, v in counts.items()
                                  if k in DRY_RECYCLABLE}},
            period="all time", kind="COMPOSITION",
        )

    def _trend(self) -> Insight | None:
        """Day over day, and only when both days carry enough items."""
        days = self.db.days()
        if len(days) < 2:
            return None

        latest, previous = days[-1], days[-2]
        now = self.db.window_summary(since=latest)
        before = self.db.window_summary(since=previous, until=latest)

        if (now["processed"] < self.min_period
                or before["processed"] < self.min_period):
            return Insight(
                headline="Not enough per-day data to compare periods",
                detail=(f"{latest} has {now['processed']} items and {previous} "
                        f"has {before['processed']}; a comparison needs "
                        f"{self.min_period} in each. No trend is being claimed."),
                evidence={"latest": latest, "latest_count": now["processed"],
                          "previous": previous,
                          "previous_count": before["processed"],
                          "threshold": self.min_period},
                period=f"{previous} to {latest}", kind="TREND",
                strength="provisional",
            )

        change = (now["processed"] - before["processed"]) / before["processed"]
        if abs(change) < self.trend_change:
            return Insight(
                headline="Throughput is steady day on day",
                detail=(f"{now['processed']} items on {latest} against "
                        f"{before['processed']} on {previous} -- a change of "
                        f"{change:+.0%}, inside the {_pct(self.trend_change)} "
                        f"band this agent treats as flat."),
                evidence={"latest": now["processed"],
                          "previous": before["processed"],
                          "change": round(change, 3)},
                period=f"{previous} to {latest}", kind="TREND",
            )

        direction = "risen" if change > 0 else "fallen"
        return Insight(
            headline=f"Throughput has {direction} {abs(change):.0%} day on day",
            detail=(f"{now['processed']} items on {latest} against "
                    f"{before['processed']} on {previous}."),
            evidence={"latest": now["processed"], "previous": before["processed"],
                      "change": round(change, 3)},
            period=f"{previous} to {latest}", kind="TREND",
        )

    def _manual_rate(self, overall: dict) -> Insight | None:
        rate = overall["manual_rate"]
        if rate < self.manual_concern:
            return None
        rules = self.db.rule_counts()
        leading = max(rules.items(), key=lambda kv: kv[1]) if rules else ("", 0)
        return Insight(
            headline=f"{_pct(rate)} of items are going to manual inspection",
            detail=(f"{overall['manual']} of {overall['processed']} items were "
                    f"held for a person, above the {_pct(self.manual_concern)} "
                    f"level worth looking at. The rung firing most often is "
                    f"{leading[0]} ({leading[1]} items)."),
            evidence={"manual": overall["manual"],
                      "processed": overall["processed"],
                      "rate": round(rate, 3), "threshold": self.manual_concern,
                      "leading_rule": leading[0], "leading_count": leading[1]},
            period="all time", kind="QUALITY",
        )

    def _confidence_association(self) -> Insight | None:
        """Two numbers that moved together -- stated as exactly that.

        The tempting sentence is "manual inspection rose BECAUSE confidence
        fell". This agent does not say because. A nudged camera, a change in
        lighting and a genuinely harder mix of waste all produce this same
        pair of movements, and nothing in the database distinguishes them.
        """
        days = self.db.days()
        if len(days) < 2:
            return None
        latest, previous = days[-1], days[-2]
        now = self.db.window_summary(since=latest)
        before = self.db.window_summary(since=previous, until=latest)
        if (now["processed"] < self.min_period
                or before["processed"] < self.min_period):
            return None

        manual_delta = now["manual_rate"] - before["manual_rate"]
        conf_delta = now["avg_confidence"] - before["avg_confidence"]

        # Only worth reporting when they moved in opposite directions and both
        # moved enough to not be noise.
        if not (manual_delta > 0.05 and conf_delta < -0.03):
            return None

        return Insight(
            headline="Manual inspection rose while detection confidence fell",
            detail=(f"Manual inspection went from {_pct(before['manual_rate'])} "
                    f"to {_pct(now['manual_rate'])} while average confidence "
                    f"went from {_pct(before['avg_confidence'])} to "
                    f"{_pct(now['avg_confidence'])}. These two moved together; "
                    f"the record does not show why. Camera position, lighting, "
                    f"objects overlapping and a genuinely harder mix of waste "
                    f"all produce this pattern."),
            evidence={"manual_before": round(before["manual_rate"], 3),
                      "manual_now": round(now["manual_rate"], 3),
                      "confidence_before": round(before["avg_confidence"], 3),
                      "confidence_now": round(now["avg_confidence"], 3)},
            period=f"{previous} to {latest}", kind="QUALITY",
        )

    def _capacity(self) -> Insight | None:
        levels = self.db.latest_bin_levels()
        if not levels:
            return None
        pressured = {name: data for name, data in levels.items()
                     if data["fraction"] >= self.capacity_pressure}
        if not pressured:
            return None
        worst = max(pressured.items(), key=lambda kv: kv[1]["fraction"])
        names = ", ".join(f"{n} {_pct(d['fraction'])}"
                          for n, d in sorted(pressured.items(),
                                             key=lambda kv: -kv[1]["fraction"]))
        return Insight(
            headline=(f"{len(pressured)} bin{'s' if len(pressured) > 1 else ''} "
                      f"at or above {_pct(self.capacity_pressure)} capacity"),
            detail=f"{names}. Highest is {worst[0]} at "
                   f"{worst[1]['level']}/{worst[1]['capacity']}.",
            evidence={name: {"level": d["level"], "capacity": d["capacity"],
                             "fraction": round(d["fraction"], 3)}
                      for name, d in pressured.items()},
            period="latest sample", kind="CAPACITY",
        )
