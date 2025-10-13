#!/bin/bash

# Usage: bash run_imageanalysis.sh

ROI_PATH="newdata/RoiSet.zip"

for folder in newdata/*; do
    for folder in newdata/*/*; do
        # Remove trailing slash for folder name
        folder="${folder%/}"
        OUT_PATH="$folder/outfinal"
        mkdir -p "$OUT_PATH"
        python3 Imageanalysis.py "$folder" "$ROI_PATH" "$OUT_PATH"
    done
done