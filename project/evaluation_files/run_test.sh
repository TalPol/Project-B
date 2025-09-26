#!/bin/bash

MODELS_DIR="./squigulator/models/redo"
OUTPUT_DIR="./squigulator/test_set/results"
DATA_DIR="./squigulator/dataset/train/pod_files/Staphylococcus_aureus_BPH2947"
REFERENCE="./squigulator/dataset/train/fasta/BPH2947__202310/reference.mmi"
LOG_FILE="./evaluation_files/redo_Staphylococcus_aureus_BPH2947_with_unmapped_mmi.txt"
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
    tail -n 10 "$TMP_LOG" >> "$LOG_FILE"
    echo -e "\n" >> "$LOG_FILE"

    echo "" >> "$LOG_FILE"

    # Clean up output directory
    rm -f "$OUTPUT_DIR"/*
    rm -f "$TMP_LOG"
done
