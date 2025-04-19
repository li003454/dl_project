# Diffusion Immunity: Project Implementation

This repository contains the implementation for the CSCI 5527 project: "Diffusion Immunity: Semantic and Biometric Attacks for Image Data Protection".

## Goal

The primary goal is to develop and evaluate a method to add imperceptible perturbations to images, particularly medical dermatology images, which achieve the following simultaneously:

1.  **Biometric Anonymization**: Degrade or prevent face detection and recognition.
2.  **Semantic Editing Resistance**: Resist unwanted edits by text-to-image diffusion models (e.g., InstructPix2Pix).
3.  **Diagnostic Feature Preservation**: Maintain the essential features needed for medical diagnosis (e.g., skin disease classification).

## Approach

We plan to implement a unified optimization pipeline (based on Projected Gradient Descent - PGD) that combines objectives targeting face recognition models and diffusion model editing mechanisms, constrained by feature preservation and perturbation imperceptibility.

## Current Progress & Implementation Details (Phase 1)

-   **Environment Setup**:
    -   Initialized a Python virtual environment (`venv`).
    -   Installed core dependencies via `pip install -r requirements.txt`, including `torch`, `torchvision`, `diffusers`, `transformers`, `opencv-python`, `pandas`, `piq`, `facenet-pytorch`, and `kagglehub`.
-   **Dataset Acquisition & Preparation**:
    -   Switched dataset direction to CelebA due to the need for facial images.
    -   Downloaded the CelebA dataset (`google/celeba`) using `kagglehub` via `scripts/download_celeba.py` into `data/raw/celeba-dataset/archive/`.
    -   Created a PyTorch Dataset loader (`data/celeba_dataset.py`) capable of:
        -   Parsing CelebA's CSV annotation files (`list_eval_partition.csv`, `list_attr_celeba.csv`).
        -   Loading images from the nested `img_align_celeba/img_align_celeba/` directory.
        -   Partitioning data into train/validation/test sets.
        -   Loading selected facial attributes.
        -   Applying standard image transformations.
-   **Baseline Model Integration**:
    -   **Face Detection/Recognition**: Implemented `models/face_recognition/facenet_wrapper.py` using `facenet-pytorch`. This wrapper loads a pre-trained MTCNN for detection and InceptionResnetV1 (VGGFace2 weights) for extracting 512-d face embeddings. Includes methods for detection, extraction, and distance calculation. Tested successfully.
    -   **Image Editing Model**: Implemented `models/diffusion/instruct_pix2pix_wrapper.py` using `diffusers`. This wrapper loads the `timbrooks/instruct-pix2pix` pipeline and provides a method to apply text-based edits to images. Tested successfully, saving example outputs.

## Project Structure

A high-level overview of the intended project structure:

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

## Next Steps (Completing Phase 1)

1.  **Integrate Baseline Model 3 (Face Attribute Classifier)**:
    -   Decide on specific attributes from CelebA to preserve (e.g., `Smiling`, `Male`, `Eyeglasses`, `Young`).
    -   Find a suitable pre-trained multi-attribute classifier OR train a simple baseline classifier (e.g., using ResNet on top of the loaded CelebA data and selected attributes from `list_attr_celeba.csv`).
    -   Create a wrapper script similar to the others (e.g., `models/attribute_classifier/classifier_wrapper.py`).
2.  **Establish Baseline Performance**:
    -   Run the clean CelebA test set through all three integrated baseline models:
        -   FaceNet: Record face detection rates and potentially average embedding distances for known identities (if identity labels are used).
        -   Attribute Classifier: Record baseline accuracy for selected attributes.
        -   InstructPix2Pix: Apply standard edits and calculate baseline image similarity metrics (SSIM, PSNR, LPIPS using `piq`) between original and edited images.
    -   Store these baseline results for later comparison with protected images.

## Getting Started

1.  Clone the repository.
2.  Create a Python virtual environment: `python3 -m venv venv`
3.  Activate the environment: `source venv/bin/activate` (or `.\venv\Scripts\activate` on Windows)
4.  Install dependencies: `pip install -r requirements.txt`
5.  Configure Kaggle API credentials (see Kaggle website and `scripts/download_celeba.py` comments).
6.  Download the dataset: `python scripts/download_celeba.py`

## Development Status

*(Tracking progress based on the REVISED implementation plan)*

- [X] Phase 1: Setup and Baselines (Partially Complete: Environment, Data, FaceNet, InstructPix2Pix Integrated)
- [ ] Phase 1: Setup and Baselines (Remaining: Attribute Classifier Integration, Baseline Evaluation)
- [ ] Phase 2: Implementation of Protection Methods
- [ ] Phase 3: Evaluation and Analysis

---

*This README reflects the updated project direction focusing on synthetic data generation.* 