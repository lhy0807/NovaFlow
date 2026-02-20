#!/bin/bash

# make sure you are in gizmo conda environment
cd wan2.1
# download the Wan2.1 I2V and FLF2V weights
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./Wan2.1-I2V-14B-720P
huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P --local-dir ./Wan2.1-FLF2V-14B-720P

cd ../tapip3d
mkdir -p checkpoints
wget https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth -O checkpoints/tapip3d_final.pth
mkdir -p third_party/megasam/Depth-Anything/checkpoints
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth -O third_party/megasam/Depth-Anything/checkpoints/depth_anything_vitl14.pth
mkdir -p third_party/megasam/cvd_opt/
cd third_party/megasam/cvd_opt/
gdown 1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM
cd ../../../

cd ../grounded_sam_2
cd checkpoints
bash download_ckpts.sh
cd ../gdino_checkpoints
bash download_ckpts.sh
cd ../../