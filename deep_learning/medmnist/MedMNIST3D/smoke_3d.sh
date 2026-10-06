#!/bin/bash
#SBATCH -q class
#SBATCH -p public
#SBATCH -t 00:20:00
#SBATCH -c 8
#SBATCH --mem=32GB
#SBATCH -G a100:1
#SBATCH -J organ3d_smoke
#SBATCH -o organ3d_smoke_%j.out
cd /scratch/jcuesta/BME/prewarmup/medmnist-experiments/MedMNIST3D
export HOME=/scratch/jcuesta/BME/prewarmup/medmnist-data
/home/jcuesta/.conda/envs/medmnist/bin/python train_and_eval_pytorch.py --data_flag organmnist3d --model_flag resnet50 --conv Conv3d --num_epochs 1 --size 28 --batch_size 32 --gpu_ids 0 --output_root ./smoke_results --run organ3d_smoke
