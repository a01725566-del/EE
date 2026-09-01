# Beginner Guide: Running This Project With Zero Setup (Google Colab)

You do not need to install Python, install Git, or use a terminal on
your own computer. Everything below happens in your web browser using
**Google Colab** (free, made by Google, runs Python for you).

Follow these steps in order. Every "code cell" instruction means:
click the `+ Code` button in Colab to add a new cell, paste in exactly
what's shown, then press **Shift + Enter** to run it and move to the
next cell.

---

## Step 1 — Download this project as a ZIP file

1. Go to: https://github.com/a01725566-del/EE
2. Near the top-left, click the branch dropdown (it might say `main`)
   and select **`claude/yolo-traffic-detection-gm6ha2`**.
3. Click the green **`<> Code`** button, then **Download ZIP**.
4. It will download as something like `EE-claude-yolo-traffic-detection-gm6ha2.zip`
   into your computer's Downloads folder. Remember this filename.

## Step 2 — Open Google Colab

1. Go to: https://colab.research.google.com
2. Sign in with a Google account if asked.
3. Click **New notebook**.

## Step 3 — Upload the ZIP into Colab

1. On the left side of the Colab screen, click the **folder icon** 📁
   (this opens the file browser for your Colab session).
2. Click the **upload icon** (a page with an up arrow) near the top of
   that file panel.
3. Choose the ZIP file you downloaded in Step 1.
4. Wait for the upload to finish (you'll see it appear in the file list).

## Step 4 — Unzip it and move into the project folder

Paste this into a code cell and run it (Shift + Enter). **Replace the
filename** if yours downloaded with a different name — check the file
panel on the left to see the exact name:

```python
!unzip -q EE-claude-yolo-traffic-detection-gm6ha2.zip
%cd EE-claude-yolo-traffic-detection-gm6ha2
```

If that runs with no errors, you're now "inside" the project folder
for the rest of the steps.

## Step 5 — Install the libraries the code needs

```python
!pip install -r requirements.txt
```

This installs OpenCV (does the image/detection work), numpy (handles
number arrays), and matplotlib (draws the chart) — all inside your
Colab session, not on your real computer.

## Step 6 — Download the YOLOv2 weights file (~194 MB, takes a minute or two)

```python
!bash download_weights.sh
```

If this fails (the original host is sometimes down), tell me and I'll
help you find a mirror — you'd then upload `yolov2.weights` into the
`model/` folder the same way you uploaded the ZIP in Step 3.

## Step 7 — Add your traffic photo

1. Get a traffic scene photo (see `sample_images/README.md` for where
   to find one and how to cite it properly for your EE).
2. In the left file panel, open the `sample_images` folder.
3. Click the upload icon and upload your photo into that folder.
4. Rename it to exactly `traffic_scene_1.jpg` (right-click the file →
   Rename), OR run this cell instead, replacing `your_photo.jpg` with
   whatever your file is actually called:

```python
!mv sample_images/your_photo.jpg sample_images/traffic_scene_1.jpg
```

## Step 8 — Run the basic detector (sanity check)

```python
!python yolo_detector.py
```

You should see text like `Found 6 object(s): - car: 91.2% confident ...`
If you see an error instead, copy the full red error message and send
it to me — don't worry, this is completely normal when setting things
up for the first time.

## Step 9 — Run the actual experiment

```python
!python run_experiment.py
```

This is the one that produces your EE data. It takes a little longer
because it runs YOLO 9 times (once per confidence threshold).

## Step 10 — Look at your results

To see the chart directly in Colab:

```python
from IPython.display import Image, display
display(Image("results/confidence_vs_detections.png"))
```

To see a labelled photo directly in Colab:

```python
from IPython.display import Image, display
display(Image("results/detected_conf_0.5.jpg"))
```

To open the results table:

```python
import pandas as pd
pd.read_csv("results/experiment_results.csv")
```

## Step 11 — Save everything to your own computer

Colab deletes your files when you close the tab, so download your
results before you finish:

```python
!zip -r results.zip results
```

Then in the left file panel, find `results.zip`, click the **⋮** (three
dots) next to it, and choose **Download**. Do the same for any
individual image you want (right-click it → Download).

---

## If something goes wrong

Copy the exact error message (the red text) and paste it back to me —
I can tell you exactly what to fix. You don't need to understand the
error yourself; that's what I'm here for.
