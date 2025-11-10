#!/bin/bash
#SBATCH --job-name=zyl_dogVScat
#SBATCH --gpus=6000ada:1
#SBATCH --time=6:00:00
#SBATCH --output=./log/job-%j.out
#SBATCH --error=./log/job-%j.err

echo "Loading modules..."
# 1. 加载 CUDA (这个非常重要, 解决 GPU 问题)
# (如果 'CUDA' 太模糊, 试试 'module load CUDA/11.8.0' 之类的具体版本)
module load CUDA 

# 2. 加载 Conda
module load Miniforge3

echo "Activating Conda environment..."
source activate mini_pro

# 检查一下 datasets 目录是否存在
echo "Checking for 'datasets/test'..."
ls -ld datasets/test
echo "--- DEBUG INFO END ---"

echo "install requirements"
pip install -r requirements.txt

echo "Starting Python script..."
# 检查 GPU 是否被正确识别
nvidia-smi 
python run_test.py

echo "Job finished."