# BME 507 - Image Processing and Analysis

Coursework and experiments completed for **BME 507: Image Processing and Analysis** at **Arizona State University**, taught by **Prof. Jianming Liang**.

This repository contains selected implementations, experiment scripts, evaluation tools, notebooks, aggregate results, and technical reports from the course. The repository is intentionally curated to exclude restricted medical datasets, patient-level data, model checkpoints, and course-provided materials.

## Contents

### Classical Image Processing
- **Otsu Thresholding** - improved implementation of Otsu's method for automatic image thresholding.
- **Active Contours / Snakes** - implementation and improvements for contour-based image segmentation.

### Warm-up Exercises
- Image reading/writing and preprocessing
- CNN-based classification
- Gender and age prediction
- Grad-CAM visualization
- Semantic lung segmentation
- Multi-organ segmentation
- Organ localization
- Training and evaluation metrics

### Deep Learning Experiments

#### Ark+
Experiment orchestration and evaluation scripts for chest X-ray classification experiments using Ark+, including:
- ChestX-ray14
- CheXpert
- MIMIC-CXR
- VinDr-CXR
- Cyclic training
- Concurrent training with equal and random sampling
- Experiment plotting and comparison utilities

The underlying Ark implementation is from the Liang Lab Ark project:
https://github.com/jlianglab/Ark

Only experiment scripts, evaluation utilities, and aggregate results are included here. Restricted datasets and derived model checkpoints are not distributed.

#### MedMNIST
Experiments with 2D and 3D MedMNIST datasets, including training scripts, HPC job scripts, and experiments exploring unified representations across multiple medical imaging tasks.

MedMNIST:
https://medmnist.com/

## Reports
Selected technical reports are included alongside the corresponding coursework. Reports contain my methodology, experiments, results, and analysis. Course handouts, lecture slides, assignment specifications, and restricted datasets are not redistributed.

## Technologies
- Python
- PyTorch
- NumPy
- Matplotlib
- OpenCV
- Jupyter / Google Colab
- SLURM / ASU SOL HPC
- Medical image analysis
- CNNs and foundation models

## Data and Privacy
This repository does **not** distribute restricted medical datasets such as MIMIC-CXR, CheXpert, or VinDr-CXR, nor does it distribute patient-level information or restricted model checkpoints. Scripts may reference dataset names or expected local paths for reproducibility, but users must obtain datasets independently and comply with their respective licenses and data-use agreements.

## Attribution and Licensing
Some experiments build upon third-party research code and datasets. Those components remain subject to their original licenses and terms. In particular, Ark/Ark+ code is associated with the Liang Lab project and MedMNIST is distributed under its own licensing terms.

This repository should not be interpreted as relicensing third-party code, datasets, or models.

## Author
**Jan Cuesta**
M.S. Robotics and Autonomous Systems, Arizona State University
