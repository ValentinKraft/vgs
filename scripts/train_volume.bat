@echo off
REM Example volume training script with common parameters for Windows

REM Replace these with your own paths
set VOLUME_PATH=_test-data_/volume.nii.gz
set MASK_PATH=_test-data_/vesselmask-float.nii.gz
set OUTPUT_DIR=output/volume-model

REM Basic training command
python train.py ^
  --model_path %OUTPUT_DIR% ^
  --mask_path %MASK_PATH% ^
  --volume_path %VOLUME_PATH% ^
  --init_n_points 5000 ^
  --save_ply_every 10 ^
  --iterations 5000 ^
  --volume_loss_type dice ^
  --volume_shape 64 64 64

REM Use the following additional parameters as needed:
REM --position_noise 0.01          # Noise for position initialization
REM --volume_loss_weight 1.0       # Weight for volume supervision loss
REM --ply_output_prefix "model"    # Custom prefix for PLY files
