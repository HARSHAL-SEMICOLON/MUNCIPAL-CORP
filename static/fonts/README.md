# The display face

Both Streamlit pages set their headings in **Summer Series**. The font is not
in this repository, because it is not mine to redistribute — so the pages ship
with a stand-in and pick up the real face the moment the file is here.

## Dropping it in

Copy the font file into this folder. Any of these work:

```
static/fonts/SummerSeries.woff2      ← best: smallest file
static/fonts/SummerSeries.woff
static/fonts/SummerSeries.ttf        ← what you usually download
static/fonts/SummerSeries.otf
```

The name only has to contain `summer` — `Summer Series Regular.ttf` is fine.
Restart the app, or just press `R`, and every heading changes. Nothing else
needs editing.

If you have a `.ttf` and want the smaller file, any web font converter will
produce the `.woff2`; it is usually about a third of the size, and the file is
inlined into the page, so the saving is on every page load.

## What you see without it

**Fredoka**, from Google Fonts — the closest free face to Summer Series, with
the same chunky rounded geometry and tall x-height. It is second in the stack,
so it hands over automatically and never needs removing. With no network at
all the page falls back again to Trebuchet MS and then the system sans, which
is plainer but never broken.

`ui.theme.using_real_font()` reports which of the two is in play.

## Why it is not committed

Display fonts are licensed per use, and a redistribution licence is a
different thing from the licence that comes with a download. `.gitignore`
keeps everything in this folder out of git except this README, so the file
stays on your machine unless you deliberately force-add it — and if your
licence does permit redistribution, that is the one-line change to make.

Body text is untouched by all of this. It stays in the system sans on purpose:
Summer Series is a display face, and a heavy display cut set at 13px across a
table of two hundred rows stops being legible and starts being texture.
