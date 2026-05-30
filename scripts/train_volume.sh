#!/bin/bash
# Example volume training script with common parameters

# Replace these with your own paths
VOLUME_PATH="_test-data_/volume.nii.gz"
MASK_PATH="_test-data_/vesselmask-float.nii.gz"
OUTPUT_DIR="output/volume-model"

# Basic training command
python train.py \
  --model_path ${OUTPUT_DIR} \
  --mask_path ${MASK_PATH} \
  --volume_path ${VOLUME_PATH} \
  --init_n_points 5000 \
  --save_ply_every 10 \
  --iterations 5000 \
  --volume_loss_type dice \
  --volume_shape 64 64 64

# Use the following additional parameters as needed:
# --position_noise 0.01          # Noise for position initialization
# --volume_loss_weight 1.0       # Weight for volume supervision loss
# --ply_output_prefix "model"    # Custom prefix for PLY files
