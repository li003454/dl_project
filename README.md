# Diffusion Immunity: Semantic and Biometric Attacks for Image Data Protection

## Goal

The primary goal is to develop and evaluate a method to add subtle perturbations to facial images (especially those with features like simulated lesions) to achieve the following simultaneously:

1.  **Biometric Anonymization**: Disrupt face detection and recognition models (e.g., FaceNet).
2.  **Semantic Editing Resistance**: Hinder unwanted edits by text-to-image diffusion models (e.g., InstructPix2Pix).
3.  **Feature Preservation**: Maintain the essential visual characteristics, particularly the simulated lesions.

## Approach

We utilize ControlNet (specifically with segmentation maps) to generate a synthetic dataset of faces with simulated lesions. This dataset serves as the foundation for developing and evaluating a unified optimization pipeline (likely PGD-based) that balances the three goals mentioned above.

## Project Structure

```
project/
├── data/
│   ├── celeba/                     # Optional: Original CelebA dataset root
│   ├── synthetic_raw/
│   │   └── base_faces/            # Preprocessed base face images (e.g., 512x512 from CelebA)
│   │       └── _prepared_ids.txt  # List of processed image IDs
│   ├── masks/
│   │   └── lesion_segmentation_maps/  # Generated segmentation masks for ControlNet
│   └── synthetic_lesion/          # Output directory for synthetic images with lesions
├── models/                       # Wrappers for baseline models (placeholders if not created yet)
│   ├── face_recognition/facenet_wrapper.py # FaceNet integration (assuming exists)
│   └── diffusion/instruct_pix2pix_wrapper.py # InstructPix2Pix integration (assuming exists)
├── scripts/
│   ├── prepare_base_faces.py      # Script for preparing and resizing base faces
│   ├── generate_lesion_masks.py   # Script for generating lesion segmentation masks
│   └── generate_synthetic_data.py # Script for generating synthetic images using ControlNet
├── venv/                          # Python virtual environment
├── .gitignore                     # Git ignore configuration
├── README.md                      # This file
└── requirements.txt               # Project dependencies
```

## Current Progress

*   ✅ **Environment Setup:** Python `venv` created; core libraries installed (`torch`, `diffusers`, `controlnet_aux`, `opencv-python`, `Pillow`, `tqdm`, etc.) via `requirements.txt`.
*   ✅ **Baseline Model Integration:** FaceNet and InstructPix2Pix wrappers implemented and tested (assuming `models/` directory structure).
*   ✅ **Synthetic Data Generation Pipeline (Phase 1 Complete):**
    *   Developed script (`prepare_base_faces.py`) to prepare base face images (resized 512x512).
    *   Developed script (`generate_lesion_masks.py`) to create corresponding segmentation masks for lesions.
    *   Developed script (`generate_synthetic_data.py`) using ControlNet (Segmentation model) to generate composite images.
    *   Successfully generated an initial batch of base faces, masks, and synthetic images.
*   ✅ **Version Control:** Project history tracked with Git; recent code, scripts, and initial data pushed to GitHub.

## Next Steps (Phase 2: Validation & Baseline Establishment)

1.  **Full Synthetic Dataset Generation:**
    *   Run `scripts/generate_synthetic_data.py` (adjust `MAX_IMAGES_TO_PROCESS` to `-1`) to generate images for all prepared base faces/masks.
    *   Monitor resource usage (VRAM/RAM, time).
2.  **Data Validation:**
    *   Visually inspect the full set of generated images in `data/synthetic_lesion/` for quality, realism, and diversity.
    *   Evaluate how accurately the generated lesions correspond to the input segmentation masks.
3.  **Establish Baselines on Synthetic Data:**
    *   Run FaceNet (`models/face_recognition/facenet_wrapper.py`) on the *clean* base faces (`data/synthetic_raw/base_faces/`) and the generated *synthetic lesion* images (`data/synthetic_lesion/`) to get baseline recognition performance and assess initial impact of lesions.
    *   Apply standard edits using InstructPix2Pix (`models/diffusion/instruct_pix2pix_wrapper.py`) to the synthetic lesion images to establish baseline editability *before* protection.
    *   *(Optional)* Develop quantitative metrics to assess lesion presence/quality in the generated images.
4.  **Refinement (If Necessary):** Based on validation and baseline results, potentially adjust ControlNet parameters (prompt, strength, steps), mask generation logic, or base faces, and regenerate data.

## Setup & Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/li003454/dl_project.git
    cd project
    ```
2.  Create and activate virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Prepare Environment:
    *   Ensure sufficient disk space for data.
    *   A CUDA-enabled GPU is highly recommended for reasonable generation times.

## Usage: Generating Synthetic Data

*(Run these scripts sequentially after setup)*

1.  **Prepare Base Faces (if not already done):**
    *   *(Optional)* Place original CelebA data appropriately or modify the script if using a different source.
    *   Run:
        ```bash
        python scripts/prepare_base_faces.py
        ```
    *   Output: Resized images in `data/synthetic_raw/base_faces/`.

2.  **Generate Lesion Masks (if not already done):**
    *   Run:
        ```bash
        python scripts/generate_lesion_masks.py
        ```
    *   Output: Segmentation masks in `data/masks/lesion_segmentation_maps/`.

3.  **Generate Synthetic Images:**
    *   *(First Run / Test)*: Check `MAX_IMAGES_TO_PROCESS` in the script (default is low). Run:
        ```bash
        python scripts/generate_synthetic_data.py
        ```
    *   *(Full Run)*: Modify `MAX_IMAGES_TO_PROCESS = -1` in `scripts/generate_synthetic_data.py`. Ensure you have time and resources. Run:
        ```bash
        python scripts/generate_synthetic_data.py
        ```
    *   Output: Final images with lesions in `data/synthetic_lesion/`. This step downloads models on the first run and requires significant computation.

## Contributing

This is a research project. Please open an issue to discuss major changes or contributions.

## License

[MIT](https://choosealicense.com/licenses/mit/) (Assumed, please update if different) 