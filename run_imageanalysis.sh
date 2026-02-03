#!/bin/bash

# Usage: bash run_imageanalysis.sh

ROI_PATH="repeats/RoiSet.zip"

for folder in repeats/*/*; do
    
    # Remove trailing slash for folder name
    folder="${folder%/}"
    OUT_PATH="$folder/out"
    mkdir -p "$OUT_PATH"
    python3 Imageanalysis.py "$folder" "$ROI_PATH" "$OUT_PATH"

done