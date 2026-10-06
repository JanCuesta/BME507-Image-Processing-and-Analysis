#!/bin/bash
#SBATCH -q class
#SBATCH -p public
#SBATCH -t 16:00:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH -G a100:1
#SBATCH -J unified18_100
#SBATCH -o unified18_100_%j.out
cd /scratch/jcuesta/BME/prewarmup/medmnist-experiments/unified_medmnist
export HOME=/scratch/jcuesta/BME/prewarmup/medmnist-data
/home/jcuesta/.conda/envs/medmnist/bin/python train_unified_all.py --epochs 100 --steps 50 --out unified18_100ep.pt
