#!/bin/bash

# Usage: bash run_imageanalysis.sh

ROI_PATH="/Users/ioannachatzitheodosiou/Desktop/scripts/Optosplit_Analysis/strain_analysis_quad/251017/RoiSet.zip"

for folder in strain_analysis_quad/251017/*; do
    
    # Remove trailing slash for folder name
    folder="${folder%/}"
    OUT_PATH="$folder/out_ml"
    mkdir -p "$OUT_PATH"
    python3 Imageanalysis_ml.py "$folder" "$ROI_PATH" "$OUT_PATH"

done