# Stage 5 — the custom detector

Everything above the detector is finished and tested. This stage removes the
one limitation the architecture cannot argue its way out of: **the current
model cannot see a battery, an aluminium can, cardboard or a plastic bag.**
Those are four of the most important items in municipal waste, and they are
not COCO classes at all, so no amount of label remapping reaches them.

```bash
python -m training.train --check
```

Validates your dataset's class names against `config/labels_waste.yaml`
without training anything. Run it first — a class the label map has never
heard of trains perfectly well and then reports `UNKNOWN` for every instance
at run time, which surfaces only as an unexplained rise in manual inspections.

---

## Dataset evaluation

Two datasets are usually suggested for this. Only one of them can be used
here, and the reason is worth understanding before you download 5 GB.

### TrashNet — **cannot be used for this project**

Not a licensing problem. A format problem: **TrashNet has no bounding
boxes.** It is an image-classification dataset — one object per image, placed
on a white posterboard, 2,527 images across six classes (glass, paper,
cardboard, plastic, metal, trash). Every image is already cropped to the
single object it contains.

A detector has to answer *where* as well as *what*, and TrashNet never says
where. You would have to re-annotate all 2,527 images with boxes yourself,
and even then the white-posterboard backgrounds look nothing like a conveyor
belt, so the model would transfer badly to the actual deployment.

Its licence is also unstated — the repository asks for a citation but names no
licence, which strictly means default copyright. Fine for a cited academic
project, worth knowing before anything is published.

**Verdict: skip.** It is a classification dataset, and this is a detection
problem.

### TACO — usable, with caveats

Trash Annotations in Context: ~1,500 images, ~4,784 annotations, COCO format
with segmentation masks (so bounding boxes come free). Litter photographed in
the wild — roads, woods, beaches — which is far closer to a real sorting
problem than a posterboard.

Licensing has three layers, and they are not the same:

| Layer | Licence |
|-------|---------|
| The repository code | MIT |
| The annotations | CC BY 4.0 |
| The images themselves | individually licensed, each under a free licence |

Attribution therefore flows per-image. Verify against the Zenodo record before
publishing anything, rather than trusting this table.

The real caveat is arithmetic: 4,784 annotations spread over 60 categories in
28 supercategories averages under 80 instances per class, and they are not
evenly spread. That is thin for detection. Use the **supercategories**, not
the 60 fine categories, and expect to collect your own images for the classes
you care about most.

**Verdict: usable as a base.** Merge with your own photographs.

### The pragmatic route

For a college project, the highest-value dataset is one you take yourself:

- Photograph the actual objects on the actual belt, under the actual lighting.
- 200–300 instances per class beats 4,000 instances of something else. A model
  trained on your conveyor will beat one trained on beaches, every time.
- Ten classes, not twenty-two. `config/labels_waste.yaml` describes the full
  target vocabulary; `training/dataset.yaml` starts with the ten that matter.
- Label with [Label Studio](https://labelstud.io) or Roboflow, export YOLO
  format.

A class with thirty examples does not fail loudly — it produces a model that
is confidently wrong about it, which is precisely what this architecture is
built to avoid. **Leave a class out rather than include it underfed.**

---

## Hardware

This machine is **CPU-only** (`torch 2.10.0+cpu`, no CUDA, 16 cores). A
100-epoch yolov8n run is roughly an hour on a modest GPU and most of a night
on CPU, so `train.py` refuses to start a CPU run without `--i-know-its-cpu`
rather than quietly occupying the laptop.

The realistic route is a free Colab or Kaggle GPU session:

1. Upload your dataset and this repository.
2. Run the same `training/train.py` there.
3. Download `best.pt` into `models/`.

The training is portable. The architecture it plugs into is already finished.

---

## Integration

When training completes, `train.py` prints the two lines to change:

```yaml
detection:
  model_path: "models/waste-v1.pt"
  label_map:  "config/labels_waste.yaml"
```

That is the entire integration. Not one line of agent code changes — the same
property the hardware seam gives the actuator, one layer up. The Material and
Classification agents read their vocabulary from the label map and have never
known which model produced a label.

Two things worth checking after the swap:

- **Re-tune the confidence bands.** `confidence.auto_sort` and
  `confidence.verify` were set for COCO. A purpose-trained model is usually
  more confident on the classes it knows and should be held to a higher bar,
  not the same one.
- **Watch the manual-inspection rate.** `python main.py --report` will tell
  you whether it moved. If it rises after the swap, the Analytics Agent will
  say so and decline to guess why — check the camera before blaming the model.
