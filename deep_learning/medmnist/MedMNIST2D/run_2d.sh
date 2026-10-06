#!/bin/bash
#SBATCH -q class
#SBATCH -p public
#SBATCH -t 02:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH -G a100:1
#SBATCH -o 2d_%x_%j.out
cd /scratch/jcuesta/BME/prewarmup/medmnist-experiments/MedMNIST2D
export HOME=/scratch/jcuesta/BME/prewarmup/medmnist-data
/home/jcuesta/.conda/envs/medmnist/bin/python train_and_eval_pytorch.py --data_flag "$1" --model_flag resnet50 --num_epochs 100 --size 28 --batch_size 128 --gpu_ids 0 --output_root "./results_runs/${1}_run${2}" --run "${1}_run$2"
