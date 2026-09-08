#!/bin/bash

# Usage: bash run_imageanalysis.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv311/bin/python"

if [[ -x "$VENV_PYTHON" ]]; then
    PYTHON_BIN="$VENV_PYTHON"
else
    PYTHON_BIN="python"
fi

ROOT_DIR="/Volumes/Untitled/strain_characterization_data/training_dataset/22L/22L_animals_902704_902767_902758_902470_ic"

if ! "$PYTHON_BIN" -c "import cv2, numpy, pandas, tifffile, read_roi, pystackreg" >/dev/null 2>&1; then
    echo "Missing Python dependencies in: $PYTHON_BIN"
    echo "Install into this interpreter, then rerun."
    exit 1
fi

find "$ROOT_DIR" -type f \( -iname "RoiSet.zip" -o -iname "*_RoiSet.zip" \) -print0 | while IFS= read -r -d '' ROI_PATH; do
    folder="$(dirname "$ROI_PATH")"
    if ! find "$folder" -maxdepth 1 -type f \( -iname "*.tif" -o -iname "*.tiff" \) | grep -q .; then
        echo "Skipping folder with no TIFF files: $folder"
        continue
    fi
    OUT_PATH="$folder/out_ml_nob"
    mkdir -p "$OUT_PATH"
    echo "Processing folder: $folder"
    echo "Using ROI: $ROI_PATH"
    "$PYTHON_BIN" "$SCRIPT_DIR/Imageanalysis_ml.py" "$folder" "$ROI_PATH" "$OUT_PATH"
done