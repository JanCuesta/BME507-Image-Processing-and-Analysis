#!/bin/bash
#SBATCH -q class
#SBATCH -p public
#SBATCH -t 00:30:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH -G a100:1
#SBATCH -J unified18_smoke
#SBATCH -o unified18_smoke_%j.out
cd /scratch/jcuesta/BME/prewarmup/medmnist-experiments/unified_medmnist
export HOME=/scratch/jcuesta/BME/prewarmup/medmnist-data
/home/jcuesta/.conda/envs/medmnist/bin/python train_unified_all.py --epochs 1 --steps 1 --out unified18_smoke.pt
