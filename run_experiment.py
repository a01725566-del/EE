"""
run_experiment.py
==================
THIS is the file that actually produces the data for my Extended Essay.

WHAT IT DOES
------------
It takes the SAME traffic photo and runs YOLOv2 on it many times,
using a different CONFIDENCE_THRESHOLD each time (and, further down,
a different INPUT SIZE too). For every run it records:
  - how many objects were detected in total
  - how many "safety-critical" objects were detected (person, bicycle,
    car, motorbike, bus, truck, traffic light, stop sign) - these are
    the object types that would most affect a robotaxi's driving
    DECISION, e.g. whether to brake for a pedestrian.
  - the individual confidence scores for those safety-critical objects

All of this gets:
  1. printed to the screen,
  2. saved as a table into results/experiment_results.csv (so I can
     paste it into a table in my EE),
  3. turned into a bar chart saved as results/confidence_vs_detections.png
     (so I can paste it in as a figure in my EE).

This file re-uses (imports) the functions from yolo_detector.py instead
of copy-pasting them, so both scripts always run YOLO in exactly the
same way - only the parameters change.

HOW TO USE
----------
1. Make sure yolo_detector.py already works on its own first (see
   README.md).
2. Change IMAGE_PATH below if needed.
3. Run:   python run_experiment.py
4. Open results/experiment_results.csv and results/confidence_vs_detections.png
"""

import os
import csv
import cv2
import matplotlib.pyplot as plt

# We import our own functions from yolo_detector.py so we don't have
# to rewrite the loading/detecting code a second time.
from yolo_detector import (
    load_class_names,
    load_yolo_network,
    detect_objects,
    draw_boxes,
    CONFIG_PATH,
    WEIGHTS_PATH,
    NAMES_PATH,
)

IMAGE_PATH = "sample_images/traffic_scene_1.jpg"

# These are the object types that matter most for a self-driving car's
# safety decisions (e.g. "brake now" vs "keep driving"). I picked these
# from the 80 COCO classes because they are the road users / road
# signals that appear in traffic scenes.
SAFETY_CRITICAL_CLASSES = [
    "person", "bicycle", "car", "motorbike", "bus", "truck",
    "traffic light", "stop sign",
]

# ---------------------------------------------------------------
# THE INDEPENDENT VARIABLE(S) OF THE EXPERIMENT
# ---------------------------------------------------------------
# Every value in this list will be tried, one at a time, as the
# CONFIDENCE_THRESHOLD, while NMS_THRESHOLD and input size are kept
# constant (this is what makes it a fair test / controlled experiment).
CONFIDENCE_VALUES_TO_TEST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Kept constant across every run (the "controlled variables"):
FIXED_NMS_THRESHOLD = 0.4
FIXED_INPUT_WIDTH = 416
FIXED_INPUT_HEIGHT = 416


def run_single_test(image, net, class_names, confidence_threshold):
    """Runs YOLO once with the given confidence_threshold and returns
    a summary dictionary describing what it found."""
    detections = detect_objects(
        image, net,
        confidence_threshold, FIXED_NMS_THRESHOLD,
        FIXED_INPUT_WIDTH, FIXED_INPUT_HEIGHT,
    )

    # Work out which of the detections are "safety-critical"
    safety_detections = [
        d for d in detections
        if class_names[d["class_id"]] in SAFETY_CRITICAL_CLASSES
    ]

    # Turn the safety-critical ones into a readable text list, e.g.
    # "person (61%), car (88%)" - useful to paste straight into the EE.
    safety_list_text = ", ".join(
        f"{class_names[d['class_id']]} ({d['confidence'] * 100:.0f}%)"
        for d in safety_detections
    )

    return {
        "confidence_threshold": confidence_threshold,
        "total_objects_detected": len(detections),
        "safety_critical_objects_detected": len(safety_detections),
        "safety_critical_detail": safety_list_text,
        "detections": detections,  # kept so we can draw/save the image too
    }


def main():
    os.makedirs("results", exist_ok=True)

    print("Loading class names, network and image...")
    class_names = load_class_names(NAMES_PATH)
    net = load_yolo_network(CONFIG_PATH, WEIGHTS_PATH)
    original_image = cv2.imread(IMAGE_PATH)
    if original_image is None:
        raise FileNotFoundError(
            f"Could not find/open '{IMAGE_PATH}'. "
            "Put a traffic photo in sample_images/ and update IMAGE_PATH."
        )

    results = []
    for confidence_value in CONFIDENCE_VALUES_TO_TEST:
        print(f"Testing CONFIDENCE_THRESHOLD = {confidence_value} ...")

        # .copy() so drawing boxes on one test doesn't affect the next test
        image_copy = original_image.copy()
        result = run_single_test(image_copy, net, class_names, confidence_value)
        results.append(result)

        # Save a labelled picture for this specific threshold, so I can
        # show a "before/after" comparison of images side by side in my EE.
        labelled = draw_boxes(image_copy, result["detections"], class_names)
        output_name = f"results/detected_conf_{confidence_value}.jpg"
        cv2.imwrite(output_name, labelled)

        print(f"  -> total objects: {result['total_objects_detected']}, "
              f"safety-critical objects: {result['safety_critical_objects_detected']}")

    save_results_to_csv(results)
    save_results_chart(results)
    print("\nDone! Check the results/ folder for the CSV, chart and images.")


def save_results_to_csv(results):
    """Writes one row per confidence-threshold test into a CSV file,
    so the numbers can be copy-pasted into a table in the EE."""
    csv_path = "results/experiment_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "confidence_threshold",
            "total_objects_detected",
            "safety_critical_objects_detected",
            "safety_critical_detail",
        ])
        for r in results:
            writer.writerow([
                r["confidence_threshold"],
                r["total_objects_detected"],
                r["safety_critical_objects_detected"],
                r["safety_critical_detail"],
            ])
    print(f"Saved table of results to {csv_path}")


def save_results_chart(results):
    """Makes a simple bar chart: confidence threshold (x-axis) against
    number of objects detected (y-axis) - one bar for "total objects"
    and one for "safety-critical objects" - and saves it as a PNG."""
    thresholds = [r["confidence_threshold"] for r in results]
    totals = [r["total_objects_detected"] for r in results]
    safety_totals = [r["safety_critical_objects_detected"] for r in results]

    # Position the two sets of bars next to each other rather than on
    # top of each other.
    x_positions = range(len(thresholds))
    bar_width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([x - bar_width / 2 for x in x_positions], totals,
            width=bar_width, label="All objects detected")
    plt.bar([x + bar_width / 2 for x in x_positions], safety_totals,
            width=bar_width, label="Safety-critical objects detected")

    plt.xticks(list(x_positions), [str(t) for t in thresholds])
    plt.xlabel("Confidence threshold")
    plt.ylabel("Number of objects detected")
    plt.title("Effect of YOLOv2 confidence threshold on detections")
    plt.legend()
    plt.tight_layout()

    chart_path = "results/confidence_vs_detections.png"
    plt.savefig(chart_path)
    print(f"Saved chart to {chart_path}")


if __name__ == "__main__":
    main()
