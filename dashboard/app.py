"""
The reporting view: history, trends and the daily record.

    streamlit run dashboard/app.py

This is the half of the dashboard that is genuinely page-shaped. The live belt
stays in the OpenCV window that `main.py` opens, because Streamlit re-runs its
script top to bottom on every interaction and fighting that at camera frame
rate is a demo that fails in the room. The two halves never talk to each other
directly -- this one only reads the SQLite file the sorting loop writes, in
read-only mode, so opening it can never disturb a running shift.

A note on the charts, since it is the kind of thing an examiner asks about:

*Single hue, not one colour per bar.* The category axis already spells out
which stream each bar is, so colouring them differently would spend the only
free visual channel on information the chart has already given. Colour is kept
for where it carries meaning on its own -- the bin chips, which mirror the
physical bins and the live window, and the status colours on alerts, which
always ship with a word beside them rather than relying on hue.

*Two charts, never two y-axes.* Items sorted per day and manual-inspection
rate per day are different scales. Putting them on one plot with two axes
would invent a correlation by choosing where the scales line up.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from core.config import Config
from core.event_bus import EventBus
from core.messages import Destination
from database.database import Database

st.set_page_config(page_title="Municipal Waste — Reporting",
                   page_icon="♻️", layout="wide")

# --- palette ---------------------------------------------------------------
# Validated with the data-viz palette checker against both surfaces. One hue
# carries every data mark; the status four are fixed and never themed.

LIGHT = {
    "hue": "#2a78d6", "ink": "#0b0b0b", "muted": "#52514e",
    "grid": "#e6e5e1", "surface": "#fcfcfb",
}
DARK = {
    "hue": "#3987e5", "ink": "#ffffff", "muted": "#c3c2b7",
    "grid": "#302f2d", "surface": "#1a1a19",
}
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

# The streams in a fixed order, so a reader finds the same row every visit and
# a filter can never repaint the survivors. These hexes match the bins in the
# live window and stand in for the physical bin colours.
STREAM_ORDER = ["RECYCLING", "ORGANIC", "E_WASTE", "HAZARDOUS", "REJECT", "MANUAL_CHECK"]
STREAM_COLOUR = {
    "RECYCLING": "#2a78d6", "ORGANIC": "#008300", "E_WASTE": "#4a3aa7",
    "HAZARDOUS": "#e34948", "REJECT": "#8a8880", "MANUAL_CHECK": "#eda100",
}
CATEGORY_ORDER = ["RECYCLABLE", "ORGANIC", "GLASS", "METAL", "PAPER",
                  "E_WASTE", "HAZARDOUS", "REJECT", "MANUAL_CHECK"]

RULE_NAMES = {
    "R1_CONTROLLER_UNAVAILABLE": "R1  actuator unavailable",
    "R2_INTRUSION": "R2  person in the zone",
    "R3_HAZARDOUS": "R3  hazardous stream",
    "R4_UNCERTAIN": "R4  uncertain",
    "R5_OVERLAP": "R5  objects overlapping",
    "R6_BIN_FULL": "R6  bin full",
    "R7_APPROVED": "R7  approved to sort",
}


def theme() -> dict:
    try:
        if st.context.theme.type == "dark":
            return DARK
    except Exception:
        pass
    try:
        if str(st.get_option("theme.base")).lower() == "dark":
            return DARK
    except Exception:
        pass
    return LIGHT


T = theme()


# --- data ------------------------------------------------------------------

@st.cache_resource
def db_path() -> Path:
    return Config.load().path("database.path", "data/waste.db")


def connect(path: Path) -> sqlite3.Connection | None:
    """Read-only, so opening the report can never disturb a running shift."""
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


@st.cache_data(ttl=5)
def fetch(sql: str, params: tuple = ()) -> list[dict]:
    conn = connect(db_path())
    if conn is None:
        return []
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


# --- chart building --------------------------------------------------------

def axis(title: str | None = None, grid: bool = False, fmt: str | None = None,
         label_limit: int = 220) -> dict:
    """Recessive chrome. The data is the loud part."""
    spec = {
        "title": title, "grid": grid, "gridColor": T["grid"], "gridWidth": 1,
        "domain": False, "ticks": False, "labelColor": T["muted"],
        "titleColor": T["muted"], "labelFontSize": 12, "titleFontSize": 11,
        "labelPadding": 6, "labelLimit": label_limit,
    }
    if fmt:
        spec["format"] = fmt
    return spec


BASE_CONFIG = {
    "background": "transparent",
    "view": {"stroke": None},
    "font": "-apple-system, Segoe UI, Roboto, sans-serif",
}


def bar_chart(rows: list[dict], label_field: str, value_field: str,
              order: list[str] | None = None, height: int = 260,
              value_format: str = ",d") -> dict:
    """Horizontal bars, one hue, with the value written on each bar.

    Direct labels are not decoration here: they are the secondary encoding
    that lets the chart be read without relying on the bar length alone.
    """
    sort = order if order else {"field": value_field, "order": "descending"}
    # Headroom on the value axis, so the direct label on the longest bar has
    # somewhere to sit instead of being clipped at the plot edge.
    top = max((r.get(value_field) or 0) for r in rows) if rows else 1
    shared = {
        "y": {"field": label_field, "type": "nominal", "sort": sort,
              "axis": axis(None)},
        "x": {"field": value_field, "type": "quantitative",
              "scale": {"domain": [0, top * 1.12]},
              "axis": axis(None, grid=True)},
    }
    return {
        "data": {"values": rows},
        "height": height,
        "layer": [
            {
                "mark": {"type": "bar", "color": T["hue"], "cornerRadiusEnd": 4,
                         "height": {"band": 0.6}},
                "encoding": {
                    **shared,
                    "tooltip": [
                        {"field": label_field, "type": "nominal", "title": " "},
                        {"field": value_field, "type": "quantitative",
                         "title": "count", "format": value_format},
                    ],
                },
            },
            {
                "mark": {"type": "text", "align": "left", "dx": 6,
                         "color": T["muted"], "fontSize": 11},
                "encoding": {
                    **shared,
                    "text": {"field": value_field, "type": "quantitative",
                             "format": value_format},
                },
            },
        ],
        "config": BASE_CONFIG,
    }


def render(spec: dict) -> None:
    st.vega_lite_chart(spec, use_container_width=True)


def chip(colour: str, label: str) -> str:
    return (f"<span style='display:inline-block;width:10px;height:10px;"
            f"border-radius:2px;background:{colour};margin-right:7px;"
            f"vertical-align:middle'></span><span style='vertical-align:middle'>"
            f"{label}</span>")


# --- page ------------------------------------------------------------------

st.title("Municipal Waste — Reporting")

path = db_path()
summary = fetch("""SELECT COUNT(*) AS processed,
                          AVG(confidence) AS avg_confidence,
                          SUM(CASE WHEN status = 'OK' THEN 1 ELSE 0 END) AS sorted_ok,
                          SUM(CASE WHEN action = 'MANUAL_CHECK' THEN 1 ELSE 0 END) AS manual
                   FROM detections""")

if not summary or not summary[0]["processed"]:
    st.info(
        f"No records yet at `{path}`.\n\n"
        "Run the sorting line first — `python main.py` — and this page will "
        "fill in. It reads the database read-only, so you can leave it open "
        "while the line runs."
    )
    st.stop()

s = summary[0]
processed = s["processed"] or 0
manual = s["manual"] or 0
sessions = fetch("SELECT * FROM sessions ORDER BY id DESC LIMIT 1")

st.caption(
    f"{path.name} · {processed:,} items on record · "
    f"last run {sessions[0]['started_at'] if sessions else 'unknown'}"
)

# --- KPI row: hero numbers, not charts -------------------------------------
a, b, c, d = st.columns(4)
a.metric("Items decided", f"{processed:,}")
b.metric("Actuated without error", f"{s['sorted_ok'] or 0:,}")
# A share of the whole, not a change over time -- so no delta arrow, which
# would imply a trend this figure does not carry.
c.metric(f"Sent to manual check ({manual / processed:.0%})" if processed
         else "Sent to manual check", f"{manual:,}")
d.metric("Average confidence", f"{(s['avg_confidence'] or 0):.0%}")

st.divider()

# --- insights and recommendations ------------------------------------------
# Both agents are built fresh on each run and given a read-only handle. The
# Planning Agent has no controller here for the same reason it has none
# anywhere: it is not allowed to do the things it suggests.


@st.cache_data(ttl=10)
def analysis() -> tuple[list, list, list]:
    from agents.analytics_agent import AnalyticsAgent
    from agents.planning_agent import PlanningAgent
    from agents.routing_agent import RoutingAgent

    cfg = Config.load()
    bus = EventBus()
    db = Database(cfg, read_only=True)
    if not db.readable:
        return [], [], []
    routing = RoutingAgent(cfg, bus)
    found = AnalyticsAgent(cfg, bus, db).analyse(announce=False)
    advice = PlanningAgent(cfg, bus, routing).recommend(found, announce=False)
    db.close()
    return found, advice, routing.table()


insights, recommendations, routes = analysis()

KIND_STYLE = {
    "CAPACITY":    ("■", STATUS["serious"]),
    "QUALITY":     ("▲", STATUS["warning"]),
    "TREND":       ("→", T["hue"]),
    "COMPOSITION": ("●", T["muted"]),
    "COVERAGE":    ("○", T["muted"]),
}

insight_col, advice_col = st.columns(2)

with insight_col:
    st.subheader("Municipal insights")
    st.caption("Computed from the record. Every finding shows its evidence, and "
               "findings are withheld when the sample is too small to support them.")
    for item in insights:
        icon, colour = KIND_STYLE.get(item.kind, ("●", T["muted"]))
        provisional = ("<span style='color:%s;font-size:11px'> · provisional</span>"
                       % T["muted"]) if item.strength != "supported" else ""
        st.markdown(
            f"<div style='padding:9px 0;border-bottom:1px solid {T['grid']}'>"
            f"<span style='color:{colour}'>{icon} {item.kind}</span>{provisional}<br>"
            f"<span style='font-size:14px;font-weight:600'>{item.headline}</span><br>"
            f"<span style='color:{T['muted']};font-size:13px'>{item.detail}</span>"
            f"</div>", unsafe_allow_html=True)
        with st.expander("Evidence"):
            st.json(item.evidence, expanded=True)
            st.caption(f"period: {item.period}")

with advice_col:
    st.subheader("Recommendations")
    st.caption("Advisory only. This system does not act on any of these — each "
               "one names the person whose decision it is.")
    for rec in recommendations:
        colour = {"HIGH": STATUS["critical"], "MEDIUM": STATUS["warning"],
                  "LOW": T["muted"]}.get(rec.priority, T["muted"])
        st.markdown(
            f"<div style='padding:9px 0;border-bottom:1px solid {T['grid']}'>"
            f"<span style='color:{colour};font-size:12px'>{rec.priority}</span> "
            f"<span style='color:{T['muted']};font-size:12px'>· {rec.owner}</span><br>"
            f"<span style='font-size:14px;font-weight:600'>{rec.headline}</span><br>"
            f"<span style='color:{T['muted']};font-size:13px'>{rec.consider}</span>"
            f"</div>", unsafe_allow_html=True)
    if not recommendations:
        st.caption("Nothing to recommend yet.")

st.divider()

# --- breakdowns ------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Waste streams")
    st.caption("Which category each item was assigned to.")
    rows = fetch("""SELECT category, COUNT(*) AS count
                    FROM detections GROUP BY category""")
    present = [c for c in CATEGORY_ORDER if any(r["category"] == c for r in rows)]
    render(bar_chart(rows, "category", "count", order=present,
                     height=max(180, 30 * len(present))))
    with st.expander("Table"):
        st.dataframe(rows, use_container_width=True, hide_index=True)

with right:
    st.subheader("Why each decision went that way")
    st.caption("The safety rung that produced the outcome. Six of the seven "
               "stop, hold or divert; only R7 sorts.")
    rows = fetch("""SELECT safety_rule, COUNT(*) AS count
                    FROM detections GROUP BY safety_rule""")
    for r in rows:
        r["rule"] = RULE_NAMES.get(r["safety_rule"], r["safety_rule"])
    order = [RULE_NAMES[k] for k in sorted(RULE_NAMES)
             if any(r["rule"] == RULE_NAMES[k] for r in rows)]
    render(bar_chart(rows, "rule", "count", order=order,
                     height=max(180, 30 * len(order))))
    with st.expander("Table"):
        st.dataframe([{"rule": r["rule"], "count": r["count"]} for r in rows],
                     use_container_width=True, hide_index=True)

st.divider()

# --- daily trend: two charts, never two y-axes -----------------------------
st.subheader("Daily record")
daily = fetch("""SELECT substr(timestamp, 1, 10) AS day,
                        COUNT(*) AS processed,
                        SUM(CASE WHEN action = 'MANUAL_CHECK' THEN 1.0 ELSE 0.0 END)
                          / COUNT(*) AS manual_rate,
                        AVG(confidence) AS avg_confidence
                 FROM detections GROUP BY day ORDER BY day""")

if len(daily) < 2:
    st.caption(
        "One day of data so far — a trend needs at least two. The figures "
        "below are today's."
    )
    e, f = st.columns(2)
    e.metric("Items today", f"{daily[0]['processed']:,}")
    f.metric("Manual-check rate today", f"{daily[0]['manual_rate']:.0%}")
else:
    e, f = st.columns(2)
    with e:
        st.caption("Items decided per day")
        render({
            "data": {"values": daily},
            "height": 220,
            "mark": {"type": "bar", "color": T["hue"], "cornerRadiusEnd": 4,
                     "width": {"band": 0.6}},
            "encoding": {
                "x": {"field": "day", "type": "ordinal", "axis": axis(None)},
                "y": {"field": "processed", "type": "quantitative",
                      "axis": axis(None, grid=True)},
                "tooltip": [{"field": "day", "type": "ordinal", "title": "day"},
                            {"field": "processed", "type": "quantitative",
                             "title": "items"}],
            },
            "config": BASE_CONFIG,
        })
    with f:
        # A separate chart, because a rate and a count share no scale.
        st.caption("Manual-inspection rate per day")
        render({
            "data": {"values": daily},
            "height": 220,
            "layer": [
                {"mark": {"type": "line", "color": T["hue"], "strokeWidth": 2,
                          "point": {"filled": True, "size": 64,
                                    "color": T["hue"]}}},
                {"mark": {"type": "point", "size": 100, "opacity": 0},
                 "encoding": {"tooltip": [
                     {"field": "day", "type": "ordinal", "title": "day"},
                     {"field": "manual_rate", "type": "quantitative",
                      "title": "manual rate", "format": ".0%"},
                 ]}},
            ],
            "encoding": {
                "x": {"field": "day", "type": "ordinal", "axis": axis(None)},
                "y": {"field": "manual_rate", "type": "quantitative",
                      "axis": axis(None, grid=True, fmt=".0%"),
                      "scale": {"domain": [0, 1]}},
            },
            "config": BASE_CONFIG,
        })
    with st.expander("Table"):
        st.dataframe(daily, use_container_width=True, hide_index=True)

st.divider()

# --- bins: small multiples, one series each --------------------------------
st.subheader("Bin fill")
bins = fetch("""SELECT bin, capacity, current_level, status, updated_at
                FROM bin_status ORDER BY id""")

if not bins:
    st.caption("No bin samples recorded yet.")
else:
    latest = {}
    for row in bins:
        latest[row["bin"]] = row

    cols = st.columns(len(STREAM_ORDER))
    for col, name in zip(cols, STREAM_ORDER):
        row = latest.get(name)
        with col:
            percent = (row["current_level"] / row["capacity"]) if row and row["capacity"] else 0
            st.markdown(chip(STREAM_COLOUR[name], name.replace("_", " ").title()),
                        unsafe_allow_html=True)
            st.progress(min(1.0, percent))
            state = row["status"] if row else "—"
            icon = "●" if state == "OK" else ("▲" if state != "FULL" else "■")
            colour = (STATUS["good"] if state == "OK"
                      else STATUS["warning"] if state != "FULL" else STATUS["critical"])
            st.markdown(
                f"<span style='color:{colour}'>{icon}</span> "
                f"<span style='color:{T['muted']};font-size:12px'>"
                f"{row['current_level'] if row else 0}/{row['capacity'] if row else 0}"
                f" · {state}</span>",
                unsafe_allow_html=True)

    # One sparkline per bin. Six lines on one plot would need six hues that
    # separate under colour-vision deficiency; six small plots need none.
    history = [dict(r, percent=(r["current_level"] / r["capacity"]) if r["capacity"] else 0)
               for r in bins]
    if len({r["updated_at"] for r in history}) > 1:
        st.caption("Fill over time")
        render({
            "data": {"values": history},
            # `columns` is a sibling of `facet`, not a property of it -- inside
            # the facet definition it is silently ignored and every plot
            # stacks into one column.
            "columns": 3,
            "facet": {"field": "bin", "type": "nominal", "sort": STREAM_ORDER,
                      "header": {"title": None, "labelColor": T["muted"],
                                 "labelFontSize": 12}},
            "spec": {
                "height": 90,
                "mark": {"type": "line", "color": T["hue"], "strokeWidth": 2},
                "encoding": {
                    "x": {"field": "updated_at", "type": "temporal", "axis": axis(None)},
                    "y": {"field": "percent", "type": "quantitative",
                          "axis": axis(None, grid=True, fmt=".0%"),
                          "scale": {"domain": [0, 1]}},
                    "tooltip": [
                        {"field": "bin", "type": "nominal", "title": "bin"},
                        {"field": "percent", "type": "quantitative",
                         "title": "fill", "format": ".0%"},
                        {"field": "updated_at", "type": "temporal", "title": "at"},
                    ],
                },
            },
            "config": BASE_CONFIG,
        })

st.divider()

# --- the record ------------------------------------------------------------
@st.cache_data(ttl=10)
def evaluation_results() -> dict:
    """Measured performance, if an evaluation has been run.

    Kept separate from the live record on purpose: the tables above describe
    what the system *did* on a shift, this describes how well it does on a
    labelled set. Conflating them would let a busy day look like accuracy.
    """
    import json
    path = ROOT / "evaluation" / "results" / "evaluation.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


ev = evaluation_results()

st.subheader("Measured performance")
if not ev:
    st.caption(
        "No evaluation on record. Everything above describes what the system "
        "*did*; nothing yet says how well it does it. Build a labelled set with "
        "`python -m evaluation.capture --label bottle` and run "
        "`python -m evaluation.evaluate` — about ten minutes."
    )
else:
    s_ = ev.get("summary", {})
    st.caption(f"{s_.get('images', 0)} labelled images · model "
               f"`{ev.get('model','?')}` · thresholds "
               f"{ev.get('thresholds', {})}")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Sort accuracy", f"{s_.get('sort_accuracy', 0):.0%}",
              help="Of the items the system chose to sort, how many reached "
                   "the correct bin. Deferrals are not counted as errors.")
    e2.metric("Coverage", f"{s_.get('coverage', 0):.0%}",
              help="Share handled without a human. Reported beside accuracy "
                   "so deferring everything cannot look like perfection.")
    e3.metric("Wrong-bin rate", f"{s_.get('wrong_bin_rate', 0):.0%}",
              help="The costly error, as a share of everything presented.")
    e4.metric("Detection accuracy", f"{s_.get('detection_accuracy', 0):.0%}",
              help="Did the model name the object. For comparison only — the "
                   "gap to sort accuracy is the architecture absorbing "
                   "perception errors.")

    sweep = ev.get("threshold_sweep", [])
    rows = [r for r in sweep if r.get("accuracy") is not None]
    if len(rows) > 1:
        st.caption("Does the confidence threshold earn its place? Higher "
                   "thresholds sort less and sort it better — read the trade.")
        long = ([{"threshold": r["threshold"], "value": r["accuracy"],
                  "measure": "accuracy"} for r in rows]
                + [{"threshold": r["threshold"], "value": r["coverage"],
                    "measure": "coverage"} for r in rows])
        render({
            "data": {"values": long},
            "height": 240,
            "layer": [
                {"mark": {"type": "line", "strokeWidth": 2,
                          "point": {"filled": True, "size": 50}},
                 "encoding": {"strokeDash": {"field": "measure", "type": "nominal",
                                             "legend": {"title": None,
                                                        "labelColor": T["muted"]}}}},
                {"mark": {"type": "point", "size": 120, "opacity": 0},
                 "encoding": {"tooltip": [
                     {"field": "measure", "type": "nominal", "title": " "},
                     {"field": "threshold", "type": "quantitative"},
                     {"field": "value", "type": "quantitative", "format": ".0%"},
                 ]}},
            ],
            "encoding": {
                "x": {"field": "threshold", "type": "quantitative",
                      "axis": axis("confidence threshold", grid=True)},
                "y": {"field": "value", "type": "quantitative",
                      "scale": {"domain": [0, 1]},
                      "axis": axis(None, grid=True, fmt=".0%")},
                "color": {"value": T["hue"]},
            },
            "config": BASE_CONFIG,
        })

    by_class = ev.get("by_class", {})
    if by_class:
        with st.expander("Per class"):
            st.dataframe(
                [{"class": k, "n": v["n"], "detected": v["object_correct"],
                  "sorted ok": v["sorted_correct"], "wrong bin": v["sorted_wrong"],
                  "deferred": v["deferred"], "avg conf": v["avg_confidence"]}
                 for k, v in sorted(by_class.items())],
                use_container_width=True, hide_index=True)

st.divider()

st.subheader("Downstream routes")
st.caption("Where each stream goes after it leaves the line. Facility fields are "
           "empty by design — the system describes the kind of destination and "
           "will not name an organisation that has not been verified. Fill them "
           "in at `config/routing.yaml`.")
if routes:
    st.dataframe(
        [{"category": r["category"], "stream": r["stream"],
          "downstream chain": r["chain"], "facility": r["facility"]}
         for r in routes],
        use_container_width=True, hide_index=True,
    )
else:
    st.caption("No routing table loaded.")

st.divider()

st.subheader("Recent decisions")
rows = fetch("""SELECT timestamp, object, confidence, material, category,
                       action, destination, status, safety_rule, reason
                FROM detections ORDER BY id DESC LIMIT 200""")
st.dataframe(
    [{"time": r["timestamp"][11:19], "object": r["object"],
      "conf": f"{r['confidence']:.0%}", "material": r["material"],
      "category": r["category"], "action": r["action"],
      "bin": r["destination"], "result": r["status"], "rule": r["safety_rule"],
      "why it went there": r["reason"]} for r in rows],
    use_container_width=True, hide_index=True, height=340,
)

st.divider()

st.subheader("Alerts")
st.caption("Identical alerts are collapsed with a count — one full bin raises "
           "one line, not one per item that met it.")
alerts = fetch("""SELECT MAX(timestamp) AS last_seen, agent, event, severity,
                         message, COUNT(*) AS times
                  FROM events
                  WHERE severity IN ('WARNING','ALERT','CRITICAL')
                  GROUP BY severity, agent, event, message
                  ORDER BY MAX(id) DESC LIMIT 40""")

if not alerts:
    st.caption("Nothing raised.")

for r in alerts:
    # Icon and word alongside the colour, never colour alone -- two of the
    # four status steps sit below 3:1 on a light surface by design.
    icon, colour = {
        "WARNING": ("▲", STATUS["warning"]),
        "ALERT": ("■", STATUS["serious"]),
        "CRITICAL": ("✕", STATUS["critical"]),
    }.get(r["severity"], ("●", T["muted"]))
    times = f" · ×{r['times']}" if r["times"] > 1 else ""
    st.markdown(
        f"<div style='padding:7px 0;border-bottom:1px solid {T['grid']}'>"
        f"<span style='color:{colour}'>{icon} {r['severity']}</span> "
        f"<span style='color:{T['muted']};font-size:12px'>"
        f"{r['last_seen'][11:19]} · {r['agent']}{times}</span><br>"
        f"<span style='font-size:13px'>{r['message'] or r['event']}</span></div>",
        unsafe_allow_html=True)
