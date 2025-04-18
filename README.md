# Diffusion Immunity: Project Implementation

This repository contains the implementation for the CSCI 5527 project: "Diffusion Immunity: Semantic and Biometric Attacks for Image Data Protection".

## Goal

The primary goal is to develop and evaluate a method to add imperceptible perturbations to images containing faces, which achieve the following simultaneously:

1.  **Biometric Anonymization**: Degrade or prevent face detection and recognition.
2.  **Semantic Editing Resistance**: Resist unwanted edits by text-to-image diffusion models (e.g., InstructPix2Pix).
3.  **Feature Preservation**: Maintain the visual integrity and key characteristics of specific regions of interest, simulated here as **artificial skin lesions** on faces, analogous to preserving diagnostic features in medical contexts.

## Approach

We aim to implement a unified optimization pipeline (likely based on Projected Gradient Descent - PGD). This pipeline will generate subtle perturbations by optimizing a loss function that combines objectives for biometric anonymization (attacking face recognition models) and semantic editing resistance (disrupting diffusion editing models), while being constrained to preserve key features (simulated lesions) and maintain overall visual quality. A core part of this approach now involves **generating a synthetic dataset** of faces with simulated lesions to provide relevant data for development and evaluation.

## Current Progress

*   ✅ **Environment Setup:** Anaconda environment created, necessary libraries installed (`torch`, `torchvision`, `diffusers`, `transformers`, `accelerate`, `controlnet_aux`, `opencv-python`, `Pillow`, `tqdm`).
*   ✅ **Data Exploration & Preparation:** Explored CelebA dataset structure.
*   ✅ **Baseline Model Integration:**
    *   FaceNet (Biometric Recognition) - Integrated for embedding extraction.
    *   InstructPix2Pix (Image Editing) - Integrated for image manipulation.
*   ✅ **Version Control:** Git repository initialized, `.gitignore` configured, project pushed to GitHub.
*   ✅ **Synthetic Data Generation (Phase 1 - Scripting Complete):**
    *   **Base Face Preparation:** Script `scripts/prepare_base_faces.py` created to select and preprocess base images (e.g., from CelebA) and save them to `data/synthetic_raw/base_faces/`. Base faces generated.
    *   **Simulated Lesion Masks:** Script `scripts/generate_lesion_masks.py` created to generate segmentation map masks for lesions and save them to `data/masks/lesion_segmentation_maps/`. Masks generated.
    *   **ControlNet Synthesis Script:** Script `scripts/generate_synthetic_data.py` created using ControlNet (Segmentation model) to generate final synthetic images with lesions based on base faces and masks, saved to `data/synthetic_lesion/`. Initial test images generated.

## Next Steps (Phase 2: Validation & Baseline)

1.  **Full Synthetic Data Generation:** Run `scripts/generate_synthetic_data.py` to generate the complete set of synthetic images with lesions.
2.  **Data Validation:**
    *   Visually inspect generated images for quality, realism, and diversity.
    *   Evaluate how well the specified lesions (from segmentation maps) are represented in the final images.
    *   Quantify diversity if necessary (e.g., attribute distribution if using CelebA base faces).
3.  **Establish Baselines on Synthetic Data:**
    *   Run FaceNet on the generated synthetic images (with lesions) to establish a baseline biometric recognition performance.
    *   Apply InstructPix2Pix (or other editing methods) to the synthetic images to understand baseline editability *before* applying protection.
    *   *(Future)* Integrate and test the facial attribute classifier on the synthetic data.
4.  **Refine Generation (If Necessary):** Based on validation results, potentially adjust prompts, ControlNet parameters, or mask generation logic and regenerate data.

## Project Structure

```
project/
├── data/
│   ├── celeba/             # CelebA dataset root (placeholder for original data location)
│   ├── synthetic_raw/
│   │   └── base_faces/     # Resized base faces (e.g., from CelebA)
│   │       └── _prepared_ids.txt # List of prepared base face IDs
│   ├── masks/
│   │   └── lesion_segmentation_maps/ # Generated segmentation masks (ControlNet input)
│   └── synthetic_lesion/   # Output directory for generated images with lesions
├── scripts/
│   ├── prepare_base_faces.py       # Script to prepare base face images
│   ├── generate_lesion_masks.py    # Script to generate lesion segmentation maps
│   └── generate_synthetic_data.py  # Script to generate final images using ControlNet
├── venv/                   # Virtual environment directory
├── .gitignore              # Specifies intentionally untracked files
├── README.md               # This file
└── requirements.txt        # Project dependencies
```

## Setup & Installation

1.  Clone the repository.
2.  Create a Python virtual environment: `python3 -m venv venv`
3.  Activate the environment: `source venv/bin/activate` (or `.\venv\Scripts\activate` on Windows)
4.  Install dependencies: `pip install -r requirements.txt`
5.  Configure Kaggle API credentials (see Kaggle website and `scripts/download_celeba.py` comments).
6.  Download the dataset: `python scripts/download_celeba.py`

## Development Status

*(Tracking progress based on the REVISED implementation plan)*

- [X] Initial Setup (Environment, Git, Initial Baselines - FaceNet, InstructPix2Pix)
- [ ] **Revised Phase 1: Synthetic Data Generation & Baselines** (In Progress)
    - [ ] Tech Selection & Setup
    - [ ] Base Face Preparation
    - [ ] Lesion Design & Mask Generation
    - [ ] Synthetic Image Generation Implementation
    - [ ] Data Validation & Lesion Baseline
- [ ] Phase 2: Implementation of Protection Methods
- [ ] Phase 3: Evaluation and Analysis

---

*This README reflects the updated project direction focusing on synthetic data generation.* 