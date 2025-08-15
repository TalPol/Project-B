#!/bin/bash

MODELS_DIR="./squigulator/models/new_dataset/redo"
OUTPUT_DIR="./squigulator/test_set/results"
DATA_DIR="./squigulator/dataset/train/pod_files/MMC234__202311"
REFERENCE="./squigulator/dataset/train/fasta/MMC234__202311/reference.mmi"
LOG_FILE="./evaluation_files/new_dataset_model_outputs_redo.txt"
TMP_LOG="./evaluation_files/tmp_model_file.txt"

# Clear previous log file if it exists
> "$LOG_FILE"

for model_dir in "$MODELS_DIR"/*/; do
    model_name=$(basename "$model_dir")

    # Add header to log file
    echo "===== Output for model: $model_name =====" >> "$LOG_FILE"

    python ./evaluation_files/get_median_identity.py run "$model_dir" \
        --directory "$DATA_DIR" \
        --reference "$REFERENCE" \
        --output_dir "$OUTPUT_DIR" \
        > "$TMP_LOG" 2>&1

    # Append only the last 10 lines (or more if needed) to the final log
    tail -n 8 "$TMP_LOG" >> "$LOG_FILE"
    echo -e "\n" >> "$LOG_FILE"

    echo "" >> "$LOG_FILE"

    # Clean up output directory
    rm -f "$OUTPUT_DIR"/*
    rm -f "$TMP_LOG"
done
