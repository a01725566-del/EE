#!/bin/bash
# download_weights.sh
#
# YOLOv2's trained "knowledge" file (yolov2.weights) is about 194 MB,
# which is too big to store in a GitHub repository, so this script
# downloads it separately onto YOUR computer.
#
# HOW TO USE:
#   1. Open a terminal in this project folder.
#   2. Run:   bash download_weights.sh
#   3. Wait for the download to finish (it is a big file, this can
#      take a few minutes depending on your internet speed).
#
# If the download fails or hangs (the original host, pjreddie.com,
# has been unreliable in recent years), search online for
# "yolov2.weights download" - you will find copies of the exact same
# official file on sites like Kaggle or GitHub mirrors. Any copy that
# is ~194 MB and named yolov2.weights will work identically here -
# just save it into the model/ folder.

mkdir -p model
echo "Downloading yolov2.weights into model/ (this is about 194 MB)..."
curl -L -o model/yolov2.weights https://pjreddie.com/media/files/yolov2.weights

echo "Done. Check that model/yolov2.weights exists and is about 194 MB:"
ls -lh model/yolov2.weights
