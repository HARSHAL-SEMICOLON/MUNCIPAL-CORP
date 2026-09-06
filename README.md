# Smart Municipal Waste Segregation System

An agentic AI system for municipal waste sorting. A camera watches a conveyor;
a chain of agents works out what each object is, what it is made of, which
waste stream it belongs to, whether it is safe to act on that conclusion, and
what to do about it — then acts, and monitors the result.

**Phase 1 (this build)** runs entirely in software: real camera, real
detection, simulated conveyor and bins.
**Phase 2 (future)** replaces one config word to drive an ESP32, servos and a
physical belt. No agent changes.

---

## Status

Stage 0 (foundations), Stage 1 (the sorting loop), Stage 3 (persistence,
monitoring, reporting) and Stage 4 (routing, analytics, planning) are complete
and tested. All ten agents are built.

Stage 5 is half done: the **perception seam** is finished, so swapping the
detection model is two config lines, and the training scaffolding is in place.
What remains is the training itself, which needs a GPU and a dataset -- see
`training/README.md`.

**Hardware is future scope.** This is a complete software system, and the
Phase 2 actuation layer is deliberately abstract rather than pending. The
controller, wire protocol, transports and Arduino sketch are written and
tested against a simulated ESP32 -- which proves the abstraction is real
rather than aspirational -- but nothing here depends on a circuit board
existing. See `firmware/README.md` when one does.

```bash
python -m tests.test_pipeline
```

```bash
python -m tests.test_database
```

```bash
python -m tests.test_analytics
```

```bash
python -m tests.test_labels
```

```bash
python -m tests.test_hardware
```

```bash
python -m tests.test_evaluation
```

172 checks between them, no camera or model required — the detector and the
database are both injectable, so the whole agent loop is testable headless in
under a second and no test touches `data/waste.db`.

Those checks prove the system is **correct**. Whether it **works** is a
different question, and it has its own answer:

```bash
python -m evaluation.evaluate
```

drives the real detector and the real agents over a labelled set of
photographs and reports sort accuracy, coverage, wrong-bin rate and a
confusion matrix. See *Measuring it* below -- the dataset takes about ten
minutes to build.

Stage 2 remains deliberately deferred: contamination detection is not reliable
with ordinary RGB vision, and this project should say so rather than pretend
around it.

---

## Quick start

```bash
pip install -r requirements.txt
```

One deliberate exception is documented at the top of `requirements.txt`:
`ultralytics-platform` is left uninstalled, because it pins `httpx>=0.28` and
provides only Ultralytics' cloud features, which this project never calls.
`pip check` will mention it; that is expected. Read the note before running
`pip install -U ultralytics`.

```bash
python main.py
```

The live window opens; the sorting loop runs at camera frame rate. For history,
trends and the daily record, in a second terminal:

```bash
streamlit run dashboard/app.py
```

It reads the same SQLite file in read-only mode, so you can leave it open while
the line runs. To fill it without holding objects up to a webcam for ten
minutes:

```bash
python -m tests.seed_demo_data --items 60 --reset
```

Point the webcam at a bottle, a banana, a phone or a book. Keys while running:

| Key | Action |
|-----|--------|
| `q` | quit |
| `s` | start / stop the belt |
| `e` | empty all bins |
| `f` | toggle injected actuator failures (see *the seam*, below) |

Other ways to run it:

```bash
python main.py --source data/clip.mp4
```

```bash
python main.py --headless 200
```

```bash
python -m tests.smoke_render
```

The last one renders one dashboard frame to `logs/dashboard_preview.png`
without needing a camera at all.

---

## The loop

```
OBSERVE → UNDERSTAND → REASON → DECIDE → ACT → MONITOR
```

| # | Agent | Question it answers | File |
|---|-------|--------------------|------|
| 1 | Vision | What objects are in the frame, and which is which across frames? | `agents/vision_agent.py` |
| 2 | Material | What is it made of — and can I actually tell? | `agents/material_agent.py` |
| 3 | Classification | Which waste stream does that put it in? | `agents/classification_agent.py` |
| 4 | Decision | What *ought* to happen to it? | `agents/decision_agent.py` |
| 5 | Safety | Is it safe to do that? | `agents/safety_agent.py` |
| 6 | Municipal Routing | Where does this stream go after the bin? | `agents/routing_agent.py` |
| 7 | Action | Make it happen, and report honestly if it didn't. | `agents/action_agent.py` |
| 8 | Monitoring | What has the shift done, and what needs attention? | `agents/monitoring_agent.py` |
| 9 | Analytics | What does the record actually show? | `agents/analytics_agent.py` |
| 10 | Planning | What should a person consider doing about it? | `agents/planning_agent.py` |

Agents never import each other. They exchange typed messages defined in
`core/messages.py`.

---

## Why this is agentic AI, not a classifier

A conventional computer-vision project is:

```
image → model → label
```

It answers one question and stops. It cannot tell you why, cannot decline to
answer, and cannot notice that it is answering badly.

This system:

1. **Perceives** continuously and tracks objects across frames
2. **Interprets** the label into a material — or refuses to
3. **Reasons** about whether an ambiguity actually matters
4. **Decides** an action, as a proposal
5. **Checks** that proposal against seven safety rules
6. **Acts** through an abstract actuator, with retries
7. **Monitors** the outcome and reports failures as failures
8. **Records** every decision with the rule that produced it

Three specific behaviours make the difference concrete:

**It knows what it does not know.** A COCO detector reports `bottle` without
distinguishing PET from glass. The Material Agent returns `UNKNOWN` with both
candidates listed rather than guessing. The Classification Agent then asks a
*different* question — does the ambiguity change where the item goes? PET and
glass are both dry recyclables headed for the same bin, so the item proceeds.
A cup that might be plastic, paper or ceramic spans two different bins, so it
goes to a human. Same uncertainty, two answers, because the consequence differs.

**It refuses to act.** Six of the seven safety rungs stop, hold or divert.
Sorting only happens when nothing objects — the last resort, not the default.

**It reports failure.** Every actuation returns a status and a latency. When
they fail, the item is held on the belt, not silently lost.

---

## The decisions that shape everything

### 1. The decision line

Detection runs every frame. **Decisions run once per object.**

Without this, a bottle in view for two seconds at 30 fps produces sixty
"decisions" — sixty database rows, sixty entries in the day's statistics, and
in Phase 2 sixty servo commands for one bottle. The system would be extremely
busy and completely wrong.

So `model.track(persist=True)` gives every object a stable id, and the agent
pipeline fires exactly once, when the object crosses a line partway across the
frame (`decision_line.position`, default 0.55). `min_frames_before_commit`
stops a single-frame flicker from committing anything.

Tested by `test_one_decision_per_object`.

### 2. The hardware seam

```
        ActionAgent
             ↓
     ActionController          ← the seam (hardware/controller.py)
        ↙          ↘
SimulationController   HardwareController
   (Phase 1)              (Phase 2)
```

```yaml
action:
  controller: simulation    # ← the entire Phase 1 → Phase 2 migration
```

Every method returns an `ActionResult` with `OK | FAILED | TIMEOUT | REFUSED`
and a latency, from the very first commit — even though the simulation cannot
fail. That is deliberate. Set `action.simulated_failure_rate: 0.6` (or press
`f` while running) and the retry-then-hold path runs against an actuator that
will not cooperate.

**The seam has now been cashed in.** `HardwareController` is implemented,
speaks a line protocol to an ESP32, and is tested against a simulated board —
and the Action Agent above it did not change by a line.
`test_action_agent_drives_hardware_unchanged` takes the retry-then-hold logic
written in Stage 1 against injected *simulation* failures and points it at a
*jamming diverter*: three attempts, then hold. No edit.

Note what is *not* retried: `REFUSED` means the controller understood and
declined — bin full, belt stopped, or a person pressing an emergency stop.
Retrying that is futile and, in the e-stop case, wrong. A jam is `FAILED`,
which is retried.

### 3. The perception seam

The hardware seam makes the actuator replaceable. This one makes the
**detector** replaceable, and it is the same idea one layer up.

```yaml
detection:
  model_path: "models/waste-v1.pt"       # ← the whole model swap
  label_map:  "config/labels_waste.yaml"
```

The COCO class names used to live inside the Material and Classification
agents. That worked while there was one model, and would have meant editing
agent logic the day a purpose-trained waste model arrived — precisely the
mistake the hardware seam exists to prevent. So the vocabulary moved to
`config/labels_*.yaml`, and no agent knows which model produced a label
(`test_agents_never_see_the_model_name` parses both agents and checks their
code, docstrings stripped).

Two maps ship. `labels_coco.yaml` is what runs today. `labels_waste.yaml`
describes a model that **does not exist yet** — and the test suite already
runs the real agents against it, confirming that a battery routes to hazardous
and an aluminium can to metal. When the weights arrive, the integration is two
lines.

A missing or malformed map degrades to "everything is UNKNOWN", which sends
items to manual inspection. That is the correct failure: a system that cannot
interpret what it sees should ask a person, not guess.

### 4. Decision proposes, Safety disposes

The Decision Agent never commands anything. It returns a proposal; the Safety
Agent holds the only path to the Action Agent.

An agent that both decides and acts has nowhere to put a rule like "never sort
an occluded object" — by the time you know it was occluded, the servo has
moved.

| Rung | Fires when | Result |
|------|-----------|--------|
| R1 | Actuator unavailable | **STOP** the belt |
| R2 | Person or animal in the sorting zone | **STOP**, manual restart required |
| R3 | Hazardous stream | **DIVERT** to hazardous, whatever the confidence |
| R4 | Confidence below threshold, or no stream assignable | **MANUAL CHECK** |
| R5 | Boxes overlap beyond `safety.overlap_iou` | **MANUAL CHECK** |
| R6 | Destination bin full | **HOLD** + alert |
| R7 | Nothing objected | **SORT** |

The ordering is the design. A hazardous item at 55% confidence must not be
handled as "low confidence, send to manual inspection" — a person would then be
picking it up by hand. It is hazardous first and uncertain second, so R3 sits
above R4.

R2 runs on *every frame*, not only when something is committed. Otherwise a
person could stand over a moving belt indefinitely as long as no waste happened
to cross the line.

---

### 5. Persistence subscribes; it is never called

The pipeline finishes an item and publishes `WASTE_RECORDED`. It does not know
that anything is counting or storing them.

```
Pipeline ──publish──> EventBus ──> MonitoringAgent ──> SQLite
                              └──> EventRecorder  ──> logs/events.jsonl
```

The Monitoring Agent owns the counters and the database writes; the sorting
path has no line of code about either. That is the claim the architecture makes
about Stage 4 — Analytics and Planning subscribe to the same events and nothing
above them moves — and Stage 3 is where it was actually tested rather than
asserted (`test_monitoring_owns_the_counters`).

Three rules govern the database, all of them tested:

**A database failure never stops the belt.** Sorting is the job; recording it is
bookkeeping. Every write is guarded; the first failure disables persistence for
the run and logs once rather than raising per item; the live counters carry on
in memory. `test_database_failure_does_not_stop_sorting` pulls the file out from
under a running line and checks the item still reaches its bin.

**Readers never block the writer.** The reporting view is a separate process
reading the same file. WAL mode makes that safe, and the view opens the file
read-only so it cannot disturb a shift in progress.

**Per-frame chatter stays out.** `WASTE_DETECTED` fires for every new object;
the database keeps decisions, verdicts, alerts and heartbeats. The full stream
is already in `logs/events.jsonl` if you want it.

Schema: `sessions`, `detections` (one row per item, with the safety rule that
produced it), `events`, `bin_status` (sampled on a timer, not written per item).

### 6. The analytics layer is built to refuse

The easiest analytics agent to write is one that always has something to say.
That agent is useless: an operator who is told "organic waste is rising" after
a two-minute run learns within a week to ignore everything it says.

So the guards do more work here than the arithmetic.

**It refuses on thin data.** Below `analytics.min_items_for_insight` the agent
returns exactly one finding — that there is not enough on record yet — and
stops. Period comparisons additionally require *both* periods to clear
`min_items_per_period`, so a full day is never compared against a lunch break.

**It reports the negative result.** An even spread across streams produces "no
single waste stream dominates", not a crown for the tallest bar.

**It reports association, not cause.** The obvious sentence is "manual
inspection rose *because* confidence fell". The agent will not write *because*.
It reports that both moved, names both numbers, and lists the alternatives — a
nudged camera, changed lighting, overlapping objects, a genuinely harder mix.
Nothing in the database distinguishes them, and pretending otherwise sends an
operator to retrain a model when the real fix was a lamp.

**Every insight carries its evidence.** An insight that cannot show its working
is indistinguishable from an invented one.

```bash
python main.py --report
```

Prints insights and recommendations from the record. No camera, no model — it
runs on a machine that has never had a webcam attached.

### 7. Planning recommends; it cannot act

The Planning Agent has no controller, no conveyor, no write path to any action
topic, and no method that changes anything. If it concludes the organic bin
needs emptying more often, the only thing that happens is a sentence in a
report addressed to whoever owns collection rounds.

That is a deliberate limit. Adjusting a collection schedule spends public money
and moves people's shifts; it is not a decision a camera on a conveyor belt
gets to make because a number crossed a threshold. Every recommendation names
an owner, and the owner is never this system — `test_planning_cannot_act`
asserts both the missing controller and that nothing but `RECOMMENDATION` ever
reaches the bus.

### 8. The routing table names no facilities

`config/routing.yaml` describes where each stream goes after the bin — "dry
collection → material recovery → authorised recycling" — and every `facility`
field is null.

That is not an unfinished feature. Naming a recycler that has not agreed to
accept a stream invents a fact about a third party, and a report saying
"batteries go to *<company>*" is worse than one saying "batteries go to an
authorised e-waste recycler", because the first can be acted on and be wrong.
Fill in the null fields with verified local data and the agent reports exactly
what you entered — the Planning Agent will even remind you that they are empty.

This is also where the category-versus-destination split finally pays off.
Glass and paper share the recycling bin but separate again downstream, so they
carry different routes. And when the Safety Agent overrides an outcome, the
route follows the override: a low-confidence banana held for inspection is not
entering the composting chain.

### A note on the charts

Worth knowing, because it is the kind of thing an examiner asks about.

*Single hue, not one colour per bar.* The category axis already names each
stream, so colouring bars differently would spend the only free visual channel
on information the chart has already given. Colour is kept for where it carries
meaning alone: the bin chips, which mirror the physical bins and the live
window, and the status colours on alerts — which always ship with a word and an
icon beside them, never hue by itself.

*Two charts, never two y-axes.* Items per day and manual-inspection rate per day
are different scales. One plot with two axes would invent a correlation by
choosing where the scales line up.

*Six small plots, not six lines.* Bin fill over time is small multiples. Six
lines on one plot would need six hues that stay distinct under colour-vision
deficiency; six plots need none.

The palette was checked with a contrast/CVD validator against both the light and
dark surfaces rather than by eye.

---

## Confidence bands

```yaml
confidence:
  auto_sort: 0.85     # ≥ 85%      sort automatically
  verify: 0.60        # 60–85%     sort, flagged as uncertain
                      # < 60%      manual inspection
```

**These are operational thresholds chosen for demonstration. They are not
validated accuracy figures and must not be presented as such.**

---

---

## Measuring it

A system that cannot say how well it works is an assertion, not a result.

```bash
python -m evaluation.capture --label bottle
```

Opens the camera, draws what the detector currently sees, and saves a frame
on SPACE. The folder name is the label, so there is nothing to annotate --
this system's output is a decision per object, not a bounding box. Twenty
frames per class, varying angle and lighting.

```bash
python -m evaluation.evaluate
```

Runs the real detector and the real agents over every image and reports:

| Metric | What it means |
|--------|---------------|
| **Sort accuracy** | of items the system *chose* to sort, how many reached the correct bin |
| **Coverage** | how much of the stream was handled without a human |
| **Wrong-bin rate** | the costly error, as a share of everything presented |
| **Detection accuracy** | did the model name the object — for comparison only |

Two choices in that table are deliberate and worth defending. **A deferral is
not an error** — refusing is a legitimate outcome here, and scoring it as
failure would reward a version that guesses. And **sort accuracy is reported
with coverage**, because a system that defers everything would otherwise
score 100%.

The gap between detection accuracy and sort accuracy is the argument for
having agents at all: a `cup` misread as a `bowl` still goes to manual check,
so the architecture absorbed a perception error.

The report ends with a threshold sweep — accuracy and coverage at each
candidate confidence value. That is what turns "0.85 is an operating
threshold, not a measured figure" from a disclaimer into a result.

Full detail in `evaluation/README.md`.


## Waste classes — and an honest limitation

The system currently runs on COCO-pretrained `yolov8n.pt`, which knows 80
everyday object classes. 31 of them are mapped to materials in
`agents/material_agent.py`:

- **Recyclable / ambiguous** — bottle, cup, bowl, wine glass, vase
- **Organic** — banana, apple, orange, broccoli, carrot, sandwich, pizza, donut, cake, hot dog
- **E-waste** — cell phone, laptop, keyboard, mouse, remote, tv, microwave, toaster, hair drier, clock
- **Paper** — book
- **Metal / sharps** — knife, scissors, fork, spoon
- **Plastic** — toothbrush

**What does not work yet, and why:** battery, aluminium can, cardboard and
plastic bag — four of the most important municipal waste items — are not COCO
classes at all. No amount of label remapping reaches them, because the model
was never trained to see them. The hazardous stream is currently reachable
only via sharps.

This is a limitation of the *model*, not the architecture, and Stage 5 has
already removed everything except the training itself. `config/labels_waste.yaml`
specifies the target vocabulary, `training/` holds the pipeline, and
`test_labels.py` runs the real agents against that vocabulary today — a battery
already routes to hazardous in the tests. What is missing is a weights file.

Raise this yourself before an examiner does, and raise it as a solved design
problem with an unfinished data problem. `python -m evaluation.evaluate` will
also quantify it for you — the miss rate on classes the model cannot see is a
measurement, not an excuse. A system that reports
`UNKNOWN` honestly for a battery is more defensible than one that confidently
calls it a mobile phone.

See `training/README.md` for the dataset evaluation. The short version: TrashNet
cannot be used at all — it is a classification dataset with no bounding boxes,
so it can never train a detector — and TACO is usable but thin, at roughly 80
annotations per class across 60 classes.

---

## Project layout

```
agents/          the ten agents (nine built)
core/            config, typed messages, event bus, routing, labels, logging
vision/          camera source, YOLO detector with tracking
simulation/      bins, conveyor, waste items, the live-window renderer
hardware/        THE SEAM: ActionController, both implementations, the
                 wire protocol, transports, and a simulated ESP32
firmware/        the Arduino sketch that runs on the real board
database/        SQLite schema, guarded writes, the reporting queries
dashboard/       the Streamlit reporting view
config/          config.yaml, routing.yaml, labels_coco.yaml, labels_waste.yaml
evaluation/      capture tool, evaluation harness, metrics
training/        custom-model pipeline + the dataset evaluation
tests/           headless tests, dashboard preview, demo seeding
pipeline.py      the closed loop and the decision line
main.py          entry point
```

Nothing in `agents/` imports anything from `hardware/` except the abstract
`ActionController`. Nothing in `agents/` knows a conveyor exists.

---

## Configuration

Everything operational lives in `config/config.yaml` — thresholds, bin
capacities, the decision-line position, the intrusion class list, the belt
speed, and the controller selection. No agent hard-codes any of them.

---

## Roadmap

| Stage | Adds | Status |
|-------|------|--------|
| 0 | Config, messages, event bus, logging, the seam | **done** |
| 1 | Vision → Material → Classification → Decision → Safety → Action, conveyor, bins, live window | **done** |
| 3 | SQLite, Monitoring Agent, event history, bin sampling, Streamlit reporting view | **done** |
| 4 | Municipal Routing Agent, Analytics Agent, Planning Agent, recommendations | **done** |
| 2 | Contamination handling — deferred; needs a sensor RGB cannot replace | |
| 5 | Evaluation harness, capture tool, metrics, threshold sweep | **done** |
| 5b | Perception seam + training pipeline done; the trained model is outstanding | in progress |
| — | Phase 2 hardware: controller, protocol, firmware written and simulated | **future scope** |

The software project is complete. What remains inside it is a trained model —
`config/labels_waste.yaml` specifies the target vocabulary, `training/` holds
the pipeline, and `test_labels.py` already runs the real agents against that
vocabulary, so a battery routes to hazardous in the tests today. What is
missing is a weights file, which needs a dataset and a GPU rather than any
further architecture.

Hardware is **future scope, not pending work**. The actuation layer is
abstract by design; the fact that a simulated ESP32 drives it with an
unchanged Action Agent is the evidence the abstraction holds. A physical
board would confirm that, not establish it.

The two halves of the dashboard are deliberately separate. The live belt is an
OpenCV window because Streamlit re-runs its script top to bottom on every
interaction, and fighting that at camera frame rate is a demo that fails in the
room. History and reports are page-shaped work that Streamlit is good at. They
share a SQLite file and nothing else.
