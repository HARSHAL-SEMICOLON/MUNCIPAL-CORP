"""
Tests for the look.

A palette is the part of a project nobody runs. Code that is wrong throws;
a colour that is wrong just sits there being slightly too pale, and the person
who notices is the examiner squinting at a projector in a bright room.

So the contrast ratios written into `ui/theme.py` are not claims, they are
assertions. Every token is recomputed here against the ground it is actually
painted on, using the WCAG relative-luminance formula, and each one has to
clear the floor its *role* needs rather than a single blanket number: body
text at 4.5:1, a bar on a chart at 3:1, a status colour lower still because it
never appears without an icon and a word next to it.

The other half of this file is the boring half, and the half that catches real
mistakes: the palette is written down twice, once in `ui/theme.py` for the CSS
and once in `.streamlit/config.toml` for the widgets Streamlit draws itself.
Two copies of a colour is a colour that will disagree with itself, so the two
are compared token by token.

Run:  python -m tests.test_theme
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.test_pipeline import PASSED, FAILED, check
from ui import theme


# --- WCAG contrast ---------------------------------------------------------

def _channel(value: int) -> float:
    c = value / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    r, g, b = (_channel(int(h[i:i + 2], 16)) for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# The floor each role has to clear, and why it is that number.
TEXT = 4.5        # WCAG AA for body text
STRONG = 7.0      # AAA, which the ink is asked for because it sets everything
GRAPHIC = 3.0     # AA for a non-text mark: a bar, a chip, a rule
LABELLED = 2.5    # a status colour, which always ships with an icon and a word


# ---------------------------------------------------------------------------

def test_text_is_readable():
    for name, tokens in (("light", theme.LIGHT), ("dark", theme.DARK)):
        ground = tokens["surface"]
        for role, floor in (("ink", STRONG), ("muted", TEXT), ("hue", TEXT)):
            ratio = contrast(tokens[role], ground)
            check(f"{name} {role} is readable on the page",
                  ratio >= floor, f"{ratio:.1f}:1, floor {floor}")

        # The card is a different ground, and the text sitting on it is the
        # same text. A palette that only works on the page background is a
        # palette that breaks the moment something is put in a box.
        for role in ("ink", "muted"):
            ratio = contrast(tokens[role], tokens["card"])
            floor = STRONG if role == "ink" else TEXT
            check(f"{name} {role} is readable on a card",
                  ratio >= floor, f"{ratio:.1f}:1, floor {floor}")


def test_marks_are_visible():
    for name, tokens in (("light", theme.LIGHT), ("dark", theme.DARK)):
        ratio = contrast(tokens["hue"], tokens["surface"])
        check(f"{name} data hue works as a bar", ratio >= GRAPHIC,
              f"{ratio:.1f}:1")

        # Not a contrast floor so much as a visibility one: a border that
        # matches its background is a border nobody can see.
        for role in ("grid", "hue_soft"):
            ratio = contrast(tokens[role], tokens["surface"])
            check(f"{name} {role} is distinguishable from the page",
                  ratio >= 1.08, f"{ratio:.2f}:1")


def test_status_colours_carry():
    pairs = ((theme.LIGHT, theme.STATUS_LIGHT, "light"),
             (theme.DARK, theme.STATUS_DARK, "dark"))
    for tokens, status, name in pairs:
        for role, colour in status.items():
            ratio = contrast(colour, tokens["surface"])
            check(f"{name} status {role} is visible",
                  ratio >= LABELLED, f"{ratio:.1f}:1, floor {LABELLED}")
        check(f"{name} status colours are four distinct values",
              len(set(status.values())) == 4)


def test_status_beat_what_they_replaced():
    """The retune was supposed to be an improvement, not a mood.

    These are the hexes the two pages used before the summer palette. Every
    new one has to be at least as visible as the one it replaced, or the
    retune traded legibility for prettiness and should be reverted.
    """
    was = {"good": "#0ca30c", "warning": "#fab219",
           "serious": "#ec835a", "critical": "#d03b3b"}
    ground = theme.LIGHT["surface"]
    for role, old in was.items():
        before = contrast(old, ground)
        after = contrast(theme.STATUS_LIGHT[role], ground)
        check(f"status {role} did not get worse", after >= before,
              f"{before:.1f}:1 -> {after:.1f}:1")


def test_bins_keep_their_meaning():
    check("both cuts describe the same six bins",
          set(theme.STREAM_LIGHT) == set(theme.STREAM_DARK) == set(theme.STREAM_ORDER),
          f"{len(theme.STREAM_ORDER)} streams")
    check("the light cut is the one exported as STREAM_COLOUR",
          theme.STREAM_COLOUR is theme.STREAM_LIGHT)

    # The municipal colours are load-bearing: blue is dry waste and green is
    # wet waste on the actual bins, so this test exists to make a future
    # restyle notice it is changing something outside its authority.
    check("recycling is still blue", theme.STREAM_LIGHT["RECYCLING"] == "#2a78d6")
    check("organic is still green", theme.STREAM_LIGHT["ORGANIC"] == "#008300")

    for name, cut in (("light", theme.STREAM_LIGHT), ("dark", theme.STREAM_DARK)):
        ground = (theme.LIGHT if name == "light" else theme.DARK)["surface"]
        for stream, colour in cut.items():
            ratio = contrast(colour, ground)
            check(f"{name} {stream} chip is visible", ratio >= GRAPHIC,
                  f"{ratio:.1f}:1")
        check(f"{name} cut has six distinct hexes", len(set(cut.values())) == 6)


def test_the_palette_is_written_down_once():
    """ui/theme.py and .streamlit/config.toml have to agree.

    They cannot be a single source, because one is read by Python at render
    time and the other by Streamlit before Python runs. So they are two
    sources, and this is the check that keeps them honest.
    """
    config = tomllib.loads(
        (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    mapping = {
        "primaryColor": "hue", "backgroundColor": "surface",
        "secondaryBackgroundColor": "card", "textColor": "ink",
        "linkColor": "hue", "borderColor": "grid",
    }
    for section, tokens in (("light", theme.LIGHT), ("dark", theme.DARK)):
        block = config["theme"][section]
        for key, token in mapping.items():
            check(f"config {section}.{key} matches the module",
                  block[key].lower() == tokens[token].lower(),
                  f"{block[key]} vs {tokens[token]}")


def test_the_display_face_degrades():
    stack = theme.DISPLAY_STACK
    check("Summer Series leads the display stack",
          stack.startswith("'Summer Series'"))
    check("something follows it", stack.count(",") >= 2,
          "a stack of one is not a stack")
    check("the stack ends in a generic family", stack.rstrip().endswith("sans-serif"))
    check("body text is not set in the display face",
          "Summer Series" not in theme.BODY_STACK)

    css = theme._font_css()
    if theme.using_real_font():
        check("the real font is inlined", css.startswith("@font-face"),
              str(theme.font_file()))
    else:
        check("the stand-in is fetched when no file is present",
              css.startswith("@import") and "Fredoka" in css)
        check("Fredoka is the named stand-in in the stack", "Fredoka" in stack)


def test_font_file_discovery():
    """Whatever the folder holds, the answer is a readable file or None."""
    path = theme.font_file()
    check("font discovery returns a real file or nothing",
          path is None or path.is_file(), str(path))
    if path is not None:
        check("the discovered file is a font Streamlit can serve",
              path.suffix.lower() in theme._FORMATS, path.suffix)


def test_opencv_conversion():
    """The demo draws boxes with cv2, which wants blue first."""
    check("hex to BGR reverses the channels",
          theme.bgr("#3B7280") == (0x80, 0x72, 0x3B))
    check("a leading hash is optional", theme.bgr("CE7671") == theme.bgr("#CE7671"))
    for token in ("hue", "muted", "ink"):
        triple = theme.bgr(theme.LIGHT[token])
        check(f"{token} converts to a valid BGR triple",
              len(triple) == 3 and all(0 <= c <= 255 for c in triple), str(triple))


def test_reference_palette_is_intact():
    """The five swatches everything else was derived from."""
    check("five swatches", len(theme.PALETTE) == 5, str(list(theme.PALETTE)))
    for name, value in theme.PALETTE.items():
        check(f"{name} is a six-digit hex",
              len(value) == 7 and value.startswith("#")
              and all(c in "0123456789ABCDEFabcdef" for c in value[1:]), value)

    # The tokens are supposed to be derived from the strip, not chosen beside
    # it. These three are taken from it unchanged, so a future edit that
    # quietly swaps one shows up here.
    check("the ink is the navy from the strip",
          theme.LIGHT["ink"].lower() == theme.PALETTE["navy"].lower())
    check("the light accent is the rose",
          theme.LIGHT["accent"].lower() == theme.PALETTE["rose"].lower())
    check("the soft fill is the mint",
          theme.LIGHT["hue_soft"].lower() == theme.PALETTE["mint"].lower())


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
