#!/bin/bash

MODELS_DIR="./models/mamba_1024/"
OUTPUT_DIR="./squigulator/models/new_dataset"

DATA_DIR="./squigulator/dataset/train/npy/train/"


for toml_file in "$MODELS_DIR"/*.toml; do
    model_name=$(basename "$toml_file" .toml)
    
    echo "Training model: $model_name"
    
    script -q -c "bonito train --epochs 2 --batch 64 --grad-accum-split 4 --quantile-grad-clip --lr 5e-5 \
        --config "$toml_file" \
        --directory "$DATA_DIR" \
        "$OUTPUT_DIR/batch_size_256/${model_name}_lr_5e-5/" -f" /dev/null

    script -q -c "bonito train --epochs 2 --batch 64 --grad-accum-split 4 --quantile-grad-clip --lr 2.5e-5 \
        --config "$toml_file" \
        --directory "$DATA_DIR" \
        "$OUTPUT_DIR/batch_size_256/${model_name}_lr_2.5e-5/" -f" /dev/null

    
done