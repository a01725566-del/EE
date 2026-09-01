# YOLOv2 Parameter Experiment — IB Extended Essay

**RQ:** To what extent does changing the parameters of the YOLO (You Only
Look Once) algorithm affect the decisions made by Apollo Go robotaxis
to safely handle traffic scenarios?

This project runs the YOLOv2 object detector on a traffic photo, then
deliberately changes YOLO's parameters and records how the detections
change. Since a self-driving car's "decision" (brake, slow down, keep
going) depends on what its object detector reports (e.g. "there is a
pedestrian 3 metres ahead"), a change in detections at the algorithm
level is treated here as a proxy for a change in the car's decision.

---

## 1. Project files

| File | What it's for |
|---|---|
| `yolo_detector.py` | Runs YOLOv2 on ONE image, ONE time. Every line is commented — read this file first to understand how YOLO works. |
| `run_experiment.py` | Runs YOLOv2 on the same image many times, changing `CONFIDENCE_THRESHOLD` each time, and saves a CSV table + bar chart + labelled images for the essay. This is the actual experiment. |
| `model/yolov2.cfg` | YOLOv2's network "blueprint" (already included). |
| `model/coco.names` | The 80 object names YOLOv2 can recognise (already included). |
| `model/yolov2.weights` | YOLOv2's trained "knowledge" — **not included** (194 MB, too big for GitHub). Download it with `download_weights.sh`. |
| `sample_images/` | Put your traffic photo(s) here. See `sample_images/README.md`. |
| `results/` | Everything the scripts generate (CSV, chart, labelled images) shows up here. |

## 2. Setup (do this once)

```bash
# 1. Install the Python libraries this project needs
pip install -r requirements.txt

# 2. Download the YOLOv2 weights file (~194 MB)
bash download_weights.sh

# 3. Add a traffic photo — see sample_images/README.md — saved as:
#    sample_images/traffic_scene_1.jpg
```

If `pip`/`bash` commands feel unfamiliar, the easiest option is to run
this whole project in **Google Colab** instead of on your own laptop:
create a new notebook, upload these files, and run the same commands
in a code cell (prefix shell commands with `!`, e.g. `!pip install -r
requirements.txt`). Colab already has most libraries pre-installed and
has no download restrictions.

## 3. Running it

```bash
# Check the basic detector works on your image first:
python yolo_detector.py

# Then run the actual experiment (this is what produces your EE data):
python run_experiment.py
```

`run_experiment.py` will print progress to the screen and create:
- `results/experiment_results.csv` — one row per confidence threshold tested
- `results/confidence_vs_detections.png` — bar chart of the results
- `results/detected_conf_0.1.jpg`, `..._0.2.jpg`, etc. — the labelled image at each threshold, useful as side-by-side figures

---

## 4. Using these results in the essay

### 4.1 How to frame the method

This experiment cannot run YOLO *inside* an actual Apollo Go car — Apollo
Go's real perception stack is proprietary, fuses camera + LiDAR + radar,
and almost certainly does not use plain YOLOv2. What this experiment
*can* do is isolate YOLOv2 itself (one component of the kind of system
Apollo Go likely relies on for camera-based object detection) and show,
with controlled variables, that its outputs are sensitive to parameter
choices. Framing this clearly and honestly in your methodology section
(as a controlled proxy experiment, not a live test of Apollo Go) is
important for EE marking criteria around methodology and reasoned
argument.

A suggested "Independent / Dependent / Controlled variables" table for
your methodology section:

| Variable type | Variable | Values used |
|---|---|---|
| Independent | `CONFIDENCE_THRESHOLD` | 0.1 → 0.9 in steps of 0.1 |
| Dependent | Number of objects detected (total, and safety-critical only) | measured from output |
| Controlled | Input image, `NMS_THRESHOLD`, input resolution, YOLO version/weights | held constant across all runs |

If you also want a second independent variable (e.g. input resolution
320/416/608, to argue about detecting *distant* pedestrians), you can
copy `run_experiment.py`, rename it, and loop over `INPUT_WIDTH` /
`INPUT_HEIGHT` the same way it currently loops over confidence — the
code structure already supports this.

### 4.2 Turning the numbers into an argument

Once you have real numbers in `results/experiment_results.csv`, the
argument usually goes:

1. **State the pattern.** e.g. "As the confidence threshold increased
   from 0.1 to 0.9, the number of detected objects fell from X to Y,
   and the pedestrian detection specifically (present at 61% confidence
   in the base run) disappeared once the threshold exceeded 0.6."
2. **Connect it to a driving decision.** A missed pedestrian at a high
   confidence threshold means the planning/control layer of a
   self-driving stack never receives "pedestrian ahead" as an input —
   so it has no reason to brake. A low confidence threshold keeps the
   pedestrian detection, but also keeps more false-positive
   detections, which could cause unnecessary/unsafe braking
   ("phantom braking") — also a safety-relevant decision.
3. **Extend to Apollo Go / real robotaxis.** Use this to discuss the
   real trade-off engineers face: safety-conscious systems are
   normally tuned to be conservative (favouring recall — catching every
   possible hazard — over precision), because a missed pedestrian is
   far more dangerous than an unnecessary brake. You can bring in
   outside sources here (Apollo/Baidu publications, YOLO papers, safety
   standards like ISO 26262 / SOTIF) to support this.
4. **Acknowledge limitations.** YOLOv2 is an older, smaller model than
   what a real 2024+ robotaxi would use; a single test image is a small
   sample; camera-only detection ignores LiDAR/radar fusion that real
   robotaxis use as a safety backup. State these clearly — EEs are
   rewarded for recognising the limits of their own method.

### 4.3 Suggested figures/tables for the essay

- **Figure:** the original photo next to 2–3 labelled versions at
  different confidence thresholds (from `results/detected_conf_*.jpg`),
  to visually show objects appearing/disappearing.
- **Figure:** `results/confidence_vs_detections.png` (the bar chart).
- **Table:** paste `results/experiment_results.csv` in directly, or
  reformat it as an actual table in your word processor.

### 4.4 Example write-up paragraph (edit with your real numbers)

> Table X shows that the number of detected objects fell as the
> confidence threshold increased from 0.1 to 0.9 (from [n] objects at
> 0.1 to [n] objects at 0.9). Critically, the pedestrian in the test
> image, initially detected at [xx]% confidence, was no longer
> detected once the threshold exceeded [xx]. Since a robotaxi's
> planning system can only react to hazards its perception system
> reports, this suggests that a poorly-chosen confidence threshold
> could cause a system built on YOLO to fail to identify a pedestrian
> that is genuinely present, directly affecting its ability to make a
> safe braking decision. This supports the argument that YOLO's
> parameters are not a neutral implementation detail, but a factor
> that can materially change a robotaxi's real-world safety behaviour.

---

## 5. Quick reference: what each parameter actually does

- **`CONFIDENCE_THRESHOLD`** — how sure YOLO must be before it reports
  a detection at all. Directly controls false negatives (missed
  hazards) vs false positives (phantom detections).
- **`NMS_THRESHOLD`** — how aggressively overlapping boxes on the same
  object get merged down to one. Mostly affects *counting* the same
  object more than once, less about missing objects entirely.
- **`INPUT_WIDTH` / `INPUT_HEIGHT`** — the resolution YOLO actually
  "looks" at. Bigger = better at spotting small/far-away objects (e.g.
  a pedestrian who is still far ahead) but slower to compute — which
  matters for a moving car that needs real-time answers.
