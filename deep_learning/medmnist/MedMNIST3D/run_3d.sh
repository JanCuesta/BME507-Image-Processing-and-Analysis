#!/bin/bash
#SBATCH -q class
#SBATCH -p public
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH -G a100:1
#SBATCH -o 3d_%x_%j.out
cd /scratch/jcuesta/BME/prewarmup/medmnist-experiments/MedMNIST3D
export HOME=/scratch/jcuesta/BME/prewarmup/medmnist-data
mkdir -p "./results_runs/${1}_run${2}" && /home/jcuesta/.conda/envs/medmnist/bin/python -u train_and_eval_pytorch.py --data_flag "$1" --model_flag resnet50 --conv Conv3d --num_epochs 100 --size 28 --batch_size 32 --gpu_ids 0 --output_root "./results_runs/${1}_run${2}" --run "${1}_run${2}"
