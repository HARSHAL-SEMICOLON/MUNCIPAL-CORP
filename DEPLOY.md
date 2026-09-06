# Deploying

## What can and cannot be deployed

Be clear about this before choosing a host, because it decides everything else.

| Part | Deployable? | Why |
|------|-------------|-----|
| **Agent demo** — upload an image, watch the chain decide | **yes** | pure request/response, no hardware |
| Live sorting line (`main.py`) | no | needs a webcam and a desktop OpenCV window |
| Shift report (`dashboard/app.py`) | technically | but `data/waste.db` is gitignored, so a hosted copy shows an empty page |

So `streamlit_app.py` is the deployment entry point, and it is the **agent
demo**, not the live loop. That is not a compromise — it is the better
showcase. The live window proves the system runs; the demo proves it
*reasons*, which is the part that distinguishes this project from a
classifier, and it works for a visitor with no camera and no install.

---

## Hugging Face Spaces — recommended

More memory than Streamlit's free tier, built for exactly this, and a Spaces
link reads well next to an ML project.

1. Create a Space: **SDK = Streamlit**, hardware = *CPU basic* (free).
2. Push this repository to the Space remote:

```bash
git remote add space https://huggingface.co/spaces/<user>/<space-name>
```

```bash
git push space main
```

3. Spaces runs `streamlit_app.py` automatically and installs from
   `requirements.txt` and `packages.txt`.

First boot takes several minutes — `torch` is a large download. Later boots
are cached.

---

## Streamlit Community Cloud — simpler, tighter

Free, connects straight to GitHub, no extra remote.

1. Go to **share.streamlit.io** → *New app*.
2. Repository `HARSHAL-SEMICOLON/MUNCIPAL-CORP`, branch `main`, main file
   `streamlit_app.py`.
3. Deploy.

The caveat is memory. The free tier is around 1 GB, and `torch` plus
`ultralytics` plus a loaded model sits close to it. It usually works with
`yolov8n` — the smallest model — but if the app restarts under load, that is
what happened, and Spaces is the answer.

---

## What is already in place for a host

**`packages.txt`** — `libgl1` and `libglib2.0-0`. OpenCV needs these on a
bare Linux image and the error without them (`ImportError: libGL.so.1`) gives
no hint that apt is the fix.

**Model weights fetched at runtime.** `models/*.pt` is gitignored — binaries
do not belong in git — so `load_system()` downloads `yolov8n.pt` on first run
and `@st.cache_resource` keeps it for the life of the container.

**Agents constructed once.** The detector and all five reasoning agents are
built inside a cached resource, so a page rerun does not reload the model.
Streamlit reruns the whole script on every interaction; without that cache
each upload would reload PyTorch.

---

## Before you deploy: add sample images

An empty demo shows nothing. Put 3–5 JPEGs in `dashboard/samples/` and they
appear in a dropdown, so a visitor sees the system decide before uploading
anything.

Pick photos that exercise different branches — `bottle.jpg` (ambiguity the
system resolves), `cup.jpg` (ambiguity it refuses), `banana.jpg` (clean
sort), `phone.jpg` (e-waste), `battery.jpg` (the honest failure). Details in
`dashboard/samples/README.md`.

These are committed, so do not put anything there you would not publish.

---

## Checks before pushing to a host

```bash
python -m tests.test_pipeline
```

```bash
streamlit run streamlit_app.py
```

- Upload an image and confirm all five agent steps render.
- Upload something the model cannot see and confirm the page explains that
  **no detection is a correct outcome**, not a crash.
- Confirm no photograph in `dashboard/samples/` shows anything private.

---

## Deploying the shift report as well

Only worth it with data behind it. `data/waste.db` is gitignored for good
reason — it holds your own run history — but a small, deliberately seeded
database makes a reasonable public demo:

```bash
python -m tests.seed_demo_data --items 60 --reset
```

Then force-add just that file, and add `dashboard/app.py` as a second page.
Every row in it comes from the real agents deciding on scripted detections,
and the `sessions` table records the source as scripted, so it is never
mistaken for a live shift.

Left undone on purpose: the demo is the stronger public artefact, and one
page that works beats two that need explaining.
