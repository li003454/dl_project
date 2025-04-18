# Diffusion Immunity: Project Implementation

This repository contains the implementation for the CSCI 5527 project: "Diffusion Immunity: Semantic and Biometric Attacks for Image Data Protection".

## Goal

The primary goal is to develop and evaluate a method to add imperceptible perturbations to images, particularly CelebA face images, which achieve the following simultaneously:

1.  **Biometric Anonymization**: Degrade or prevent face detection and recognition.
2.  **Semantic Editing Resistance**: Resist unwanted edits by text-to-image diffusion models (e.g., InstructPix2Pix).
3.  **Feature Preservation**: Maintain essential facial attributes (e.g., `Smiling`, `Male`, `Eyeglasses`) or overall image quality.

## Approach

We aim to implement a unified optimization pipeline (likely based on Projected Gradient Descent - PGD). This pipeline will generate subtle perturbations by optimizing a loss function that combines objectives for biometric anonymization (attacking face recognition models) and semantic editing resistance (disrupting diffusion editing models), while being constrained to preserve key facial attributes and maintain visual quality.

## Current Progress & Implementation Details (Phase 1 - Part 1: Foundations)

We have successfully laid the groundwork for the project:

-   **Environment Setup**: 
    -   **What**: Created a Python `venv` and installed core libraries (`torch`, `diffusers`, `facenet-pytorch`, `pandas`, `piq`, etc.) via `requirements.txt`.
    -   **Significance**: Ensures a consistent and reproducible development environment with all necessary tools available.
-   **Dataset Acquisition & Preparation**: 
    -   **What**: Switched to the CelebA dataset, downloaded it using `kagglehub` (`scripts/download_celeba.py`), and implemented a PyTorch `Dataset` loader (`data/celeba_dataset.py`) for images and attributes.
    -   **Significance**: Provides the essential data source (images with faces and attributes) and the mechanism to efficiently load and preprocess it for experiments.
-   **Baseline Model 1 - Face Recognition (`models/face_recognition/facenet_wrapper.py`)**: 
    -   **What**: Integrated the FaceNet model (MTCNN for detection, InceptionResnetV1 for embeddings).
    -   **Significance**: This model serves two roles: 
        1.  It's a primary **target** for our biometric anonymization goal (we want our perturbations to fool this model).
        2.  It's a crucial **evaluation tool** to measure how well our protection methods actually anonymize faces.
-   **Baseline Model 2 - Image Editing (`models/diffusion/instruct_pix2pix_wrapper.py`)**: 
    -   **What**: Integrated the InstructPix2Pix model for text-guided image editing.
    -   **Significance**: This model also serves two roles:
        1.  It's the **target** for our semantic editing resistance goal (we want perturbations to disrupt its editing ability).
        2.  It's the **evaluation tool** to measure how resistant our protected images are to unwanted edits.
-   **Version Control**: 
    -   **What**: Initialized Git, configured `.gitignore`, and pushed the initial codebase to GitHub.
    -   **Significance**: Ensures code safety, tracks changes, and facilitates collaboration.

*In summary, the project infrastructure is set up, data is accessible, and the core "adversary" models (for both biometric and semantic attacks) are integrated and ready for use as attack targets and evaluation benchmarks.* 

## Project Structure

A high-level overview of the current project structure:

```
diffusion_immunity/
├── configs/              # Configuration files for models and experiments
├── data/                 # Data loading, preprocessing, and storage
│   ├── raw/celeba-dataset/archive/ # Raw CelebA files
│   └── celeba_dataset.py      # Dataset loader class
├── evaluation/           # Evaluation metrics, scripts, and results storage
├── models/               # Baseline models (Face, Diffusion, Diagnosis/Attribute)
│   ├── face_recognition/facenet_wrapper.py # FaceNet integration
│   └── diffusion/instruct_pix2pix_wrapper.py # InstructPix2Pix integration
├── protection_methods/   # Implementation of protection techniques (FaceLock, EditShield, Unified)
├── notebooks/            # Experimental notebooks
├── output/               # Directory for generated outputs (e.g., edited images)
├── scripts/              # Main runnable scripts (download, training, evaluation, perturbation)
│   └── download_celeba.py   # Script to download dataset
├── utils/                # Utility functions
├── requirements.txt      # Project dependencies
└── README.md             # This file
```
*(Structure will evolve as project progresses)*

## Next Steps (Completing Phase 1 & Moving to Phase 2)

The immediate next steps focus on completing the baseline setup before developing the core protection methods:

1.  **Integrate Baseline Model 3 (Face Attribute Classifier)**:
    -   **Goal**: Obtain a model that can predict facial attributes (e.g., `Smiling`, `Male`, `Eyeglasses`). This model will act as the **evaluation tool** for our "Feature Preservation" objective.
    -   **Task**: Find a pre-trained model or train a simple classifier using `CelebADataset` and `list_attr_celeba.csv`. Create a wrapper (e.g., `models/attribute_classifier/classifier_wrapper.py`).
2.  **Establish Baseline Performance**: 
    -   **Goal**: Quantify the performance of the *unprotected* data on the three baseline models.
    -   **Task**: Run the clean CelebA test set through FaceNet (measure detection/recognition), InstructPix2Pix (measure edit impact via SSIM/LPIPS), and the Attribute Classifier (measure attribute prediction accuracy). Store these results as the benchmark.

**Following Phase 1:**

3.  **Phase 2: Implement Protection Methods**: 
    -   **Goal**: Develop the core perturbation generation algorithms.
    -   **Task**: Implement individual attacks (FaceLock-like, EditShield-like) and the proposed unified multi-objective attack based on PGD.
4.  **Phase 3: Evaluation and Analysis**: 
    -   **Goal**: Assess how well the developed protection methods achieve the project's three main goals compared to the baseline.
    -   **Task**: Apply protections to the test set, re-evaluate using the baseline models and metrics, analyze trade-offs, and draw conclusions.

## Getting Started

1.  Clone the repository.
2.  Create a Python virtual environment: `python3 -m venv venv`
3.  Activate the environment: `source venv/bin/activate` (or `.\venv\Scripts\activate` on Windows)
4.  Install dependencies: `pip install -r requirements.txt`
5.  Configure Kaggle API credentials (see Kaggle website and `scripts/download_celeba.py` comments).
6.  Download the dataset: `python scripts/download_celeba.py`

## Development Status

*(Tracking progress based on the implementation plan phases)*

- [X] Phase 1: Setup and Baselines (Part 1: Environment, Data, FaceNet, InstructPix2Pix Integrated)
- [ ] Phase 1: Setup and Baselines (Part 2: Attribute Classifier Integration, Baseline Evaluation)
- [ ] Phase 2: Implementation of Protection Methods
- [ ] Phase 3: Evaluation and Analysis

---

*This README provides a starting point and will be updated throughout the project.* 