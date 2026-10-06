#!/bin/bash
#SBATCH -q class
#SBATCH -p public
#SBATCH -t 00:15:00
#SBATCH -c 4
#SBATCH --mem=24GB
#SBATCH -G a100:1
#SBATCH -J unified_smoke
#SBATCH -o unified_smoke_%j.out
cd /scratch/jcuesta/BME/prewarmup/medmnist-experiments/unified_medmnist
export HOME=/scratch/jcuesta/BME/prewarmup/medmnist-data
/home/jcuesta/.conda/envs/medmnist/bin/python train_unified.py --epochs 1 --blood_n 64 --organ_n 32 --out unified_smoke.pt
