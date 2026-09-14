"""
The look: one palette, one display face, applied to both Streamlit surfaces.

This is a module rather than a block of CSS pasted into each app because the
demo and the reporting view had drifted into two copies of the same palette
dictionary. Two copies of a palette is a palette that will disagree with
itself the first time somebody edits one of them, and the disagreement shows
up as a chart in one shade of teal and a bin chip in another.

The display face is **Summer Series**. It is not a Google font and it is not
redistributable here, so it loads from `static/fonts/` when the file is
present and Fredoka stands in when it is not. Both resolve to the same family
name, so nothing downstream has to know which one it got: drop the real file
in and every heading changes on the next rerun, with no other edit anywhere.

Body text deliberately stays in a plain system sans. Summer Series is a
display face, wonderful at 34px across the top of a page and tiring at 13px
across a table of two hundred decisions.

Two things the summer palette is *not* allowed to touch:

**The six bin hues.** Those are not a design choice. Blue for dry recyclables
and green for wet waste is what the Solid Waste Management Rules put on the
physical bins, so the summer palette does not get a vote. What they do get is
a second cut for the dark theme, because the indigo that reads perfectly well
on paper sits at 2.0:1 on a dark ground and all but disappears.

A caveat worth writing down: the live OpenCV window uses the same six
*meanings* but six different hexes -- its recycling is #2e98b0 against this
page's #2a78d6, and so on for the other five. The comment that used to sit
over this table claimed the two matched. They never have. Reconciling them
means editing `simulation/renderer.py` as well, which is a change to the live
window and not to these pages.

**The four status steps.** Green, amber, orange and red carry meaning a
seaside palette cannot express. They are retuned rather than replaced, warmed
and desaturated to sit beside the rest, and every one of them now clears its
old contrast ratio on both grounds. They still never travel without an icon
and a word beside them, because two of the four sit below 3:1 on a light
surface even after the retune.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "static" / "fonts"


# --- the reference palette -------------------------------------------------
# Sampled from the summer strip: rose, blush, mint, teal, navy. Everything
# below is derived from these five rather than invented alongside them.

PALETTE = {
    "rose":  "#CE7671",
    "blush": "#E9A49E",
    "mint":  "#BAD3CB",
    "teal":  "#4A8996",
    "navy":  "#2A3752",
}

# The ratios in these comments are against that theme's page ground, measured
# rather than assumed. tests/test_theme.py recomputes every one of them and
# fails if an edit here drops a token below the floor its role needs.

LIGHT = {
    "hue":      "#3B7280",   # 5.1:1  data marks, links, the one data colour
    "hue_soft": "#BAD3CB",   #        fills and rails, never text
    "ink":      "#2A3752",   # 11.2:1 the navy, straight from the strip
    "muted":    "#5C6B85",   # 5.1:1  secondary text, still passes AA
    "grid":     "#DDE6E2",   #        rules and gridlines
    "card":     "#EFF4F2",   #        raised surfaces
    "surface":  "#FBF8F7",   #        the page itself, warm rather than white
    "accent":   "#CE7671",   #        the rose, spent sparingly
}

DARK = {
    "hue":      "#7FB8C4",   # 7.6:1
    "hue_soft": "#2F4A55",
    "ink":      "#F3EFED",   # 14.6:1
    "muted":    "#A7B3C6",   # 7.9:1
    "grid":     "#333E56",
    "card":     "#222C41",
    "surface":  "#171E2B",
    "accent":   "#E9A49E",
}

STATUS_LIGHT = {
    "good":     "#2F8F5B",   # 3.8:1, up from 3.2:1
    "warning":  "#C98A1E",   # 2.8:1, up from 1.7:1
    "serious":  "#CE7671",   # 3.1:1, the rose doing real work
    "critical": "#B23A3A",   # 5.6:1
}

STATUS_DARK = {
    "good":     "#5FBE8A",   # 7.3:1
    "warning":  "#E7B45C",   # 8.8:1
    "serious":  "#E9A49E",   # 8.2:1, the blush
    "critical": "#E2726C",   # 5.5:1
}

# The physical bins, in a fixed order so a reader finds the same row every
# visit. The hues are the municipal ones and the summer palette leaves them
# alone; the dark cut is the same six hues lifted until they clear 3:1 on a
# dark ground, which the originals do not.
STREAM_ORDER = ["RECYCLING", "ORGANIC", "E_WASTE", "HAZARDOUS", "REJECT",
                "MANUAL_CHECK"]
STREAM_LIGHT = {
    "RECYCLING": "#2a78d6", "ORGANIC": "#008300", "E_WASTE": "#4a3aa7",
    "HAZARDOUS": "#e34948", "REJECT": "#8a8880",
    # Darkened from #eda100, which sat at 2.0:1 on a near-white page and was
    # effectively invisible as a chip -- and worse, this is the colour the demo
    # sets the outcome word in, so "MANUAL CHECK" was the one verdict a reader
    # could not read. Safe to change: manual check is the system's own stream
    # for handing an item to a person, not one of the municipal bin colours.
    "MANUAL_CHECK": "#B07C10",
}
STREAM_DARK = {
    "RECYCLING": "#6FA8EC", "ORGANIC": "#4FB35A", "E_WASTE": "#9186E0",
    "HAZARDOUS": "#F0817F", "REJECT": "#B3B0A7", "MANUAL_CHECK": "#F0BC4A",
}
# Kept as a name because both pages import it and the light cut is the one a
# reader means when they talk about "the bin colours".
STREAM_COLOUR = STREAM_LIGHT

DISPLAY_STACK = ("'Summer Series', 'Fredoka', 'Trebuchet MS', 'Segoe UI', "
                 "system-ui, sans-serif")
BODY_STACK = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
              "'Helvetica Neue', Arial, sans-serif")
MONO_STACK = "'JetBrains Mono', 'SFMono-Regular', Consolas, monospace"


# --- the display face ------------------------------------------------------

_FORMATS = {
    ".woff2": ("font/woff2", "woff2"),
    ".woff":  ("font/woff", "woff"),
    ".ttf":   ("font/ttf", "truetype"),
    ".otf":   ("font/otf", "opentype"),
}


def font_file() -> Path | None:
    """The Summer Series file, if somebody has dropped one in.

    Matched by name rather than by an exact filename, because a font that has
    been bought or downloaded arrives called any of a dozen things. Smallest
    format wins, woff2 ahead of ttf: the file is inlined into the page, and a
    90KB ttf costs roughly three times what the same face costs as woff2.
    """
    if not FONT_DIR.is_dir():
        return None
    files = [p for p in FONT_DIR.iterdir() if p.suffix.lower() in _FORMATS]
    named = [p for p in files if "summer" in p.stem.lower()]
    candidates = named or files
    if not candidates:
        return None
    rank = list(_FORMATS)
    return sorted(candidates, key=lambda p: rank.index(p.suffix.lower()))[0]


@st.cache_data(show_spinner=False)
def _face(path_str: str, mtime: float, size: int) -> str:
    """The font inlined as a data URI.

    Inlined rather than served out of `static/`, because static serving
    depends on the directory the app was launched from and on a config flag a
    host is free to override. A data URI has neither dependency: if the file
    was readable when the page rendered, the font is in the page.

    Keyed on mtime and size as well as path, so replacing the file with a
    different cut of the same name actually takes effect.
    """
    path = Path(path_str)
    mime, fmt = _FORMATS[path.suffix.lower()]
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return ("@font-face{font-family:'Summer Series';font-style:normal;"
            "font-weight:400 900;font-display:swap;"
            "src:url(data:" + mime + ";base64," + payload + ") "
            "format('" + fmt + "')}")


def _font_css() -> str:
    path = font_file()
    if path is not None:
        try:
            stat = path.stat()
            return _face(str(path), stat.st_mtime, stat.st_size)
        except OSError:
            pass
    # No file yet. Fredoka is the closest free face to Summer Series, same
    # chunky rounded geometry and the same tall x-height, and it sits second
    # in the stack so it hands over the moment the real file appears.
    return ("@import url('https://fonts.googleapis.com/css2?"
            "family=Fredoka:wght@500;600;700&display=swap');")


def using_real_font() -> bool:
    """Whether the page is showing Summer Series or the stand-in."""
    return font_file() is not None


# --- resolution ------------------------------------------------------------

def is_dark() -> bool:
    """Which theme the browser is actually showing.

    Two sources, because they disagree in different situations: the browser
    context knows what the viewer picked from the menu, and the config option
    knows what the app was launched with. Neither exists outside a running
    script, hence the belt and braces.
    """
    try:
        return st.context.theme.type == "dark"
    except Exception:
        pass
    try:
        return str(st.get_option("theme.base")).lower() == "dark"
    except Exception:
        return False


def resolve() -> tuple[dict, dict]:
    """Tokens and status colours for the theme in play."""
    return (DARK, STATUS_DARK) if is_dark() else (LIGHT, STATUS_LIGHT)


def streams() -> dict:
    """Bin colours for the theme in play. Same six hues, two cuts."""
    return STREAM_DARK if is_dark() else STREAM_LIGHT


def _css(T: dict) -> str:
    """One stylesheet, with the tokens already substituted.

    Values are baked in rather than left to a `prefers-color-scheme` block,
    because Streamlit's theme is a setting inside the app and not a browser
    preference: a media query would light up the dark CSS on a light page
    every time the two disagreed.
    """
    return f"""
{_font_css()}

:root {{
  --display:{DISPLAY_STACK};
  --body:{BODY_STACK};
  --mono:{MONO_STACK};
  --ink:{T['ink']};
  --muted:{T['muted']};
  --hue:{T['hue']};
  --hue-soft:{T['hue_soft']};
  --grid:{T['grid']};
  --card:{T['card']};
  --surface:{T['surface']};
  --accent:{T['accent']};
}}

.stApp {{ background:var(--surface); font-family:var(--body); }}
.stAppHeader, header[data-testid="stHeader"] {{ background:transparent; }}

.stMainBlockContainer, .block-container {{
  padding-top:2.4rem; padding-bottom:4rem; max-width:1320px;
}}

/* Headings carry the display face. Sizes are set here rather than left to
   Streamlit's defaults, because a face this heavy at Streamlit's h1 size
   overpowers everything underneath it. */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
  font-family:var(--display); color:var(--ink); letter-spacing:.005em;
}}
.stApp h1 {{
  font-size:2.3rem; line-height:1.1; text-transform:uppercase;
  letter-spacing:.02em; margin:0 0 .1rem; padding:0;
}}
.stApp h2 {{ font-size:1.38rem; line-height:1.2; margin:.4rem 0 .2rem; }}
.stApp h3 {{ font-size:1.1rem; line-height:1.25; margin:.3rem 0 .2rem; }}

.stApp p, .stApp li, .stApp label {{ color:var(--ink); }}
.stApp a {{ color:var(--hue); text-underline-offset:2px; }}

/* The eyebrow above a page title. Same treatment as the monospace tags in the
   agent trace, so the two read as one family of labels. */
.pmc-kicker {{
  font-family:var(--display); font-size:.72rem; letter-spacing:.22em;
  text-transform:uppercase; color:var(--hue); margin:0 0 .3rem;
}}
.pmc-lede {{
  font-size:1rem; line-height:1.55; color:var(--muted);
  max-width:76ch; margin:.55rem 0 0;
}}
.pmc-rule {{
  height:5px; width:76px; border-radius:3px; margin:.9rem 0 1.3rem;
  background:linear-gradient(90deg,{PALETTE['teal']} 0%,
             {PALETTE['mint']} 45%,{PALETTE['blush']} 72%,
             {PALETTE['rose']} 100%);
}}

/* KPI tiles. Streamlit sets a bare number on the page ground; a card with a
   rule down its left edge groups the four into a row that reads as one
   object rather than four loose figures. */
[data-testid="stMetric"] {{
  background:var(--card); border:1px solid var(--grid);
  border-left:3px solid var(--hue); border-radius:10px;
  padding:.8rem 1rem .9rem;
}}
[data-testid="stMetricLabel"] p {{
  font-size:.72rem !important; letter-spacing:.09em; text-transform:uppercase;
  color:var(--muted) !important; font-weight:600;
}}
[data-testid="stMetricValue"] {{
  font-family:var(--display); font-size:2rem; color:var(--ink);
  line-height:1.15;
}}

[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
.stCaption, .stCaption p {{ color:var(--muted) !important; }}

hr, [data-testid="stDivider"] hr {{ border-color:var(--grid); opacity:1; }}

[data-testid="stExpander"] details {{
  border:1px solid var(--grid); border-radius:10px; background:var(--card);
}}
[data-testid="stExpander"] summary {{ font-weight:600; color:var(--ink); }}

[data-testid="stFileUploaderDropzone"] {{
  border:1.5px dashed var(--hue-soft); background:var(--card);
  border-radius:10px;
}}

.stProgress > div > div > div {{ background-color:var(--grid); }}
.stProgress > div > div > div > div {{ background-color:var(--hue); }}

code, kbd {{ font-family:var(--mono); }}

/* The agent trace and the outcome card. Styled here instead of inline so the
   two apps cannot drift apart again. */
.pmc-step {{ padding:.7rem 0; border-bottom:1px solid var(--grid); }}
.pmc-tag {{
  font-family:var(--mono); font-size:.68rem; letter-spacing:.1em;
  color:var(--muted); text-transform:uppercase;
}}
.pmc-q {{ font-size:.8rem; color:var(--muted); }}
.pmc-a {{ font-size:1.05rem; font-weight:650; color:var(--ink); }}
.pmc-d {{ font-size:.82rem; color:var(--muted); }}
.pmc-outcome {{
  margin-top:1.1rem; padding:1.05rem 1.2rem; background:var(--card);
  border:1px solid var(--grid); border-left:5px solid var(--hue);
  border-radius:12px;
}}
.pmc-outcome-word {{
  font-family:var(--display); font-size:1.85rem; line-height:1.12;
  text-transform:uppercase; letter-spacing:.015em;
}}
.pmc-row {{ padding:.55rem 0; border-bottom:1px solid var(--grid); }}
.pmc-row-head {{ font-size:.88rem; font-weight:650; color:var(--ink); }}
.pmc-row-body {{ font-size:.82rem; color:var(--muted); }}
"""


def apply() -> tuple[dict, dict]:
    """Resolve the theme, inject the stylesheet, hand back the tokens.

    Call once, immediately after `st.set_page_config`. Returns the colour
    tokens and the status four, in that order, so a page can keep using them
    for the inline styles and chart specs that CSS cannot reach.
    """
    T, status = resolve()
    st.markdown(f"<style>{_css(T)}</style>", unsafe_allow_html=True)
    return T, status


# --- small shared pieces ---------------------------------------------------

def hero(title: str, kicker: str, lede: str | None = None) -> None:
    """A page title that announces itself, set in the display face."""
    st.markdown(
        f"<div class='pmc-kicker'>{kicker}</div>"
        f"<h1>{title}</h1>"
        + (f"<p class='pmc-lede'>{lede}</p>" if lede else "")
        + "<div class='pmc-rule'></div>",
        unsafe_allow_html=True,
    )


def bgr(hex_colour: str) -> tuple[int, int, int]:
    """A palette token as OpenCV wants it.

    The demo draws its bounding boxes with cv2, which takes blue first. Doing
    the conversion here means the box on the photograph is literally the same
    token as the rule under the heading beside it, instead of a hand-written
    triple that has to be remembered about when the palette changes.
    """
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def chip(colour: str, label: str) -> str:
    return (f"<span style='display:inline-block;width:10px;height:10px;"
            f"border-radius:3px;background:{colour};margin-right:7px;"
            f"vertical-align:middle'></span><span style='vertical-align:middle'>"
            f"{label}</span>")


def chart_config(T: dict) -> dict:
    """Vega-Lite chrome. The body face, not the display face: chart labels are
    small, and a heavy display cut at that size turns into texture."""
    return {
        "background": "transparent",
        "view": {"stroke": None},
        "font": BODY_STACK,
        "axis": {"labelColor": T["muted"], "titleColor": T["muted"]},
        "legend": {"labelColor": T["muted"], "titleColor": T["muted"]},
    }
