#!/bin/bash
#SBATCH -q class
#SBATCH -p public
#SBATCH -t 00:30:00
#SBATCH -c 4
#SBATCH --mem=24GB
#SBATCH -G a100:1
#SBATCH -J unified_5ep
#SBATCH -o unified_5ep_%j.out
cd /scratch/jcuesta/BME/prewarmup/medmnist-experiments/unified_medmnist
export HOME=/scratch/jcuesta/BME/prewarmup/medmnist-data
/home/jcuesta/.conda/envs/medmnist/bin/python train_unified.py --epochs 5 --blood_n 256 --organ_n 128 --out unified_5ep.pt
