"""
yolo_detector.py
=================
This is the "core" script for my Extended Essay experiment.

WHAT THIS FILE DOES
--------------------
It loads ONE traffic photo, runs the YOLOv2 object detector on it, and
draws a box around every object YOLO found (cars, people, traffic
lights, etc.), together with the label and how confident YOLO is.

WHY THIS MATTERS FOR MY RQ
---------------------------
My research question asks how CHANGING YOLO's parameters affects the
"decisions" a robotaxi like Apollo Go would make (e.g. "is there a
pedestrian here or not?"). This file contains the three parameters I
am investigating, all in one place near the top, so I can change them
and immediately see if YOLO's output changes.

This file only runs the detector ONCE, with whatever settings are
typed in below. The actual experiment (running it many times with
different settings and recording the results) happens in
run_experiment.py, which imports and reuses the functions written here
so I don't have to copy-paste code.

HOW TO USE THIS FILE
----------------------
1. Put a traffic photo inside the sample_images/ folder.
2. Get the model files into the model/ folder (see README.md -
   model/yolov2.cfg and model/coco.names are already included in this
   project, but model/yolov2.weights is too big for GitHub, so you
   have to download it yourself - instructions are in README.md).
3. Change IMAGE_PATH below to the name of your photo.
4. Run this file:   python yolo_detector.py
5. Look inside the results/ folder for the labelled output image.
"""

# ---------------------------------------------------------------
# STEP 1: IMPORT THE LIBRARIES WE NEED
# ---------------------------------------------------------------
import cv2          # OpenCV - loads/saves images AND can run a YOLO
                     # (Darknet) neural network for us, so we don't have
                     # to write the neural network maths ourselves.
import numpy as np  # numpy - lets us work with the arrays (lists of
                     # numbers) that OpenCV's YOLO output comes in.
import os            # os - just used to make sure the results/ folder
                     # exists before we try to save a file into it.


# ---------------------------------------------------------------
# STEP 2: THE PARAMETERS I AM INVESTIGATING FOR MY EE
# ---------------------------------------------------------------
# These three numbers are the "independent variables" of my experiment.
# I will keep changing them (in run_experiment.py) and record what
# happens to YOLO's detections.

# CONFIDENCE_THRESHOLD: a number between 0 and 1.
# YOLO looks at the image and, for every object it *thinks* it sees,
# it gives itself a confidence score (e.g. 73% sure this is a "car").
# If that score is BELOW this threshold, we throw the detection away.
#   -> Lower threshold  = YOLO keeps more detections (but some may be
#      wrong / false alarms).
#   -> Higher threshold = YOLO keeps fewer detections (but is more
#      certain about the ones it keeps - it might miss real objects).
CONFIDENCE_THRESHOLD = 0.5

# NMS_THRESHOLD: "Non-Maximum Suppression" threshold, also 0 to 1.
# YOLO often draws several overlapping boxes on the SAME object. NMS
# is the clean-up step that deletes the extra, overlapping boxes and
# keeps only the best one. This number controls how much two boxes
# are allowed to overlap before one of them gets deleted.
#   -> Lower value  = more boxes get deleted (stricter clean-up).
#   -> Higher value = more overlapping boxes are allowed to survive.
NMS_THRESHOLD = 0.4

# INPUT_WIDTH / INPUT_HEIGHT: YOLO does not look at the photo at its
# original size. It first resizes (shrinks/stretches) it to a square
# of this many pixels. This has to be a multiple of 32 for YOLOv2.
#   -> Smaller (e.g. 320)  = faster, but small/far-away objects
#      (like a distant pedestrian) become harder to detect.
#   -> Larger (e.g. 608)   = slower, but small/far-away objects are
#      easier to detect.
INPUT_WIDTH = 416
INPUT_HEIGHT = 416


# ---------------------------------------------------------------
# STEP 3: WHERE ALL THE FILES ARE
# ---------------------------------------------------------------
CONFIG_PATH = "model/yolov2.cfg"        # the network's "blueprint"
WEIGHTS_PATH = "model/yolov2.weights"   # the network's trained "knowledge"
NAMES_PATH = "model/coco.names"         # list of the 80 object names YOLO knows

IMAGE_PATH = "sample_images/traffic_scene_1.jpg"   # <- change to your photo
OUTPUT_PATH = "results/traffic_scene_1_detected.jpg"  # where we save the result


# ---------------------------------------------------------------
# STEP 4: A FEW SMALL "HELPER" FUNCTIONS
# ---------------------------------------------------------------
# Each function below does ONE small job. run_experiment.py will import
# and reuse these exact same functions, so both scripts behave
# identically and I'm not copy-pasting code.

def load_class_names(names_path):
    """Read coco.names and return it as a plain Python list of strings,
    e.g. ["person", "bicycle", "car", ...]. The line number in the file
    matches the class ID number that YOLO outputs."""
    with open(names_path, "r") as f:
        # .read().strip() removes any blank line at the end of the file
        # .split("\n") turns the text into one list entry per line
        class_names = f.read().strip().split("\n")
    return class_names


def load_yolo_network(config_path, weights_path):
    """Load the YOLOv2 network into OpenCV using the cfg (blueprint)
    and weights (trained knowledge) files, and return it."""
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    return net


def get_output_layer_names(net):
    """YOLO's network has many layers, but we only want the numbers
    that come out of the FINAL detection layer(s). This function asks
    OpenCV which layer names those are."""
    all_layer_names = net.getLayerNames()
    # net.getUnconnectedOutLayers() gives the index numbers of the
    # output layers. We use those index numbers to look up their names.
    output_indexes = net.getUnconnectedOutLayers()
    output_layer_names = [all_layer_names[i - 1] for i in output_indexes.flatten()]
    return output_layer_names


def detect_objects(image, net, confidence_threshold, nms_threshold,
                    input_width, input_height):
    """
    Runs YOLO on ONE image and returns a clean Python list of detections.

    Each detection in the returned list is a dictionary that looks like:
        {"class_id": 2, "confidence": 0.87, "box": [x, y, w, h]}

    This is the single most important function in the whole project -
    everything about my experiment comes from calling this function
    with different parameter values.
    """
    image_height, image_width = image.shape[:2]

    # --- 4a. Turn the image into a "blob" that YOLO can understand ---
    # A blob is just the image, resized to (input_width, input_height),
    # with its pixel colour values scaled from the normal 0-255 range
    # down to 0.0-1.0 (that's what the 1/255 does), and with the
    # colour channel order swapped from OpenCV's BGR to YOLO's RGB
    # (that's what swapRB=True does).
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (input_width, input_height),
        swapRB=True, crop=False
    )

    # --- 4b. Feed the blob into the network and run it ---
    net.setInput(blob)
    output_layer_names = get_output_layer_names(net)
    layer_outputs = net.forward(output_layer_names)
    # layer_outputs now holds YOLO's raw predictions: for every possible
    # box, a set of numbers describing where it is, how confident YOLO
    # is, and which of the 80 classes it thinks the object belongs to.

    # --- 4c. Go through every raw prediction and keep the useful ones ---
    boxes = []          # will hold [x, y, w, h] for each kept box
    confidences = []    # will hold the confidence score for each kept box
    class_ids = []       # will hold which class (car/person/etc) for each box

    for output in layer_outputs:
        for detection in output:
            # The first 5 numbers in `detection` are box info (we don't
            # need them yet); everything AFTER that is one confidence
            # score per class (80 numbers, one per COCO class).
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)     # the class with the highest score
            confidence = class_scores[class_id]     # that highest score

            # This is exactly where CONFIDENCE_THRESHOLD gets used:
            if confidence > confidence_threshold:
                # YOLO gives box positions as FRACTIONS of the image
                # size (0.0-1.0), so we multiply by the real image
                # width/height to get actual pixel coordinates.
                box_center_x = int(detection[0] * image_width)
                box_center_y = int(detection[1] * image_height)
                box_w = int(detection[2] * image_width)
                box_h = int(detection[3] * image_height)

                # OpenCV wants the TOP-LEFT corner of the box, not the
                # centre, so we shift by half the width/height.
                x = int(box_center_x - box_w / 2)
                y = int(box_center_y - box_h / 2)

                boxes.append([x, y, box_w, box_h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # --- 4d. Remove duplicate/overlapping boxes (Non-Max Suppression) ---
    # This is exactly where NMS_THRESHOLD gets used.
    kept_indexes = cv2.dnn.NMSBoxes(
        boxes, confidences, confidence_threshold, nms_threshold
    )

    # --- 4e. Build the clean list of detections we actually keep ---
    detections = []
    if len(kept_indexes) > 0:
        for i in kept_indexes.flatten():
            detections.append({
                "class_id": class_ids[i],
                "confidence": confidences[i],
                "box": boxes[i],
            })
    return detections


def draw_boxes(image, detections, class_names):
    """Draws a rectangle + text label for every detection, straight
    onto the image. Returns the same image, now with boxes drawn on it."""
    for det in detections:
        x, y, w, h = det["box"]
        label = class_names[det["class_id"]]
        confidence = det["confidence"]
        text = f"{label} {confidence * 100:.0f}%"

        # cv2.rectangle draws the box: image, top-left, bottom-right, colour (BGR), thickness
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText draws the text label just above the box
        cv2.putText(image, text, (x, max(y - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image


# ---------------------------------------------------------------
# STEP 5: RUN EVERYTHING (only happens when you run THIS file directly,
# not when run_experiment.py imports functions from it)
# ---------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)  # create results/ if it's missing

    print("Loading class names...")
    class_names = load_class_names(NAMES_PATH)

    print("Loading YOLOv2 network (this can take a few seconds)...")
    net = load_yolo_network(CONFIG_PATH, WEIGHTS_PATH)

    print(f"Reading image: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(
            f"Could not find/open '{IMAGE_PATH}'. "
            "Put a traffic photo in sample_images/ and update IMAGE_PATH."
        )

    print("Running YOLOv2 detection...")
    detections = detect_objects(
        image, net,
        CONFIDENCE_THRESHOLD, NMS_THRESHOLD,
        INPUT_WIDTH, INPUT_HEIGHT
    )

    print(f"Found {len(detections)} object(s):")
    for det in detections:
        name = class_names[det["class_id"]]
        print(f"  - {name}: {det['confidence'] * 100:.1f}% confident")

    labelled_image = draw_boxes(image, detections, class_names)
    cv2.imwrite(OUTPUT_PATH, labelled_image)
    print(f"Saved labelled image to {OUTPUT_PATH}")
