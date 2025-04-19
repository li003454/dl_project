import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
from tqdm import tqdm
import numpy as np # Might be needed for image processing if any

def load_models(base_model_id="runwayml/stable-diffusion-v1-5", 
                  controlnet_model_id="lllyasviel/control_v11p_sd15_seg", 
                  device=None):
    """Loads the base Stable Diffusion model and the ControlNet model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for ControlNet generation")

    try:
        print(f"Loading ControlNet model: {controlnet_model_id}")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id, 
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )
        
        print(f"Loading base Stable Diffusion model: {base_model_id}")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            safety_checker=None # Disable safety checker if needed
        )
        
        # Use a potentially faster scheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        print("Models loaded successfully.")
        return pipe, device
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Ensure model IDs are correct, you have an internet connection, and sufficient VRAM/RAM.")
        return None, None

def generate_synthetic_images(pipe, device, base_face_dir, mask_dir, output_dir, 
                                num_images = -1, # -1 means process all found
                                prompt="photo of a face, detailed skin texture, realistic lighting, high quality", 
                                negative_prompt="blurry, deformed, unrealistic, multiple faces, watermark, text, signature, cartoon, drawing, illustration, sketch",
                                num_inference_steps=20, 
                                guidance_scale=7.5,
                                controlnet_conditioning_scale=0.8 # ControlNet strength
                                ):
    """
    Generates synthetic images using ControlNet based on segmentation masks.

    Args:
        pipe: The loaded StableDiffusionControlNetPipeline.
        device: The torch device.
        base_face_dir (str): Directory containing base face images (used for filenames).
        mask_dir (str): Directory containing the segmentation masks.
        output_dir (str): Directory to save the generated synthetic images.
        num_images (int): Max number of images to process (-1 for all).
        prompt (str): Text prompt for generation.
        negative_prompt (str): Negative text prompt.
        num_inference_steps (int): Number of diffusion steps.
        guidance_scale (float): Guidance scale for the prompt.
        controlnet_conditioning_scale (float): How much influence the ControlNet has.
    """
    print(f"Starting synthetic image generation...")
    print(f"Reading base face list from: {base_face_dir}")
    print(f"Reading masks from: {mask_dir}")
    print(f"Saving outputs to: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Get the list of base face filenames (to match with masks)
    try:
        ids_list_path = os.path.join(base_face_dir, '_prepared_ids.txt')
        if os.path.exists(ids_list_path):
             with open(ids_list_path, 'r') as f:
                 base_face_files = [line.strip() for line in f if line.strip()]
             print(f"Found list of {len(base_face_files)} prepared IDs.")
        else:
            print("Warning: '_prepared_ids.txt' not found in base face dir. Listing mask directory instead.")
            # Fallback: list files in mask dir and derive base names
            mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith('.png')]
            # Derive base filenames (remove .png extension)
            base_face_files = [os.path.splitext(f)[0] for f in mask_files]
            if not base_face_files:
                 print(f"Error: No mask files found in {mask_dir}")
                 return
            print(f"Found {len(base_face_files)} mask files.")

    except Exception as e:
        print(f"Error accessing input directories or ID list: {e}")
        return

    processed_count = 0
    files_to_process = base_face_files[:num_images] if num_images > 0 else base_face_files

    for base_filename in tqdm(files_to_process, desc="Generating synthetic images"):
        mask_filename = base_filename + ".png" # Assume mask has .png extension
        mask_path = os.path.join(mask_dir, mask_filename)
        output_path = os.path.join(output_dir, base_filename) # Output filename same as base face

        if not os.path.exists(mask_path):
            print(f"Warning: Mask file not found for {base_filename} at {mask_path}. Skipping.")
            continue

        try:
            # Load the segmentation mask
            control_image = Image.open(mask_path).convert("RGB")
            # Ensure mask size matches expected input size (e.g., 512x512)
            # Optional: Add resize if needed, but masks should be generated at correct size
            # control_image = control_image.resize((512, 512))

            # Generate image using ControlNet pipeline
            # The control_image (segmentation map) is passed as 'image' argument
            generator = torch.Generator(device=device).manual_seed(-1) # Use random seed
            output_image = pipe(
                prompt,
                image=control_image, # Pass the segmentation map here!
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator
            ).images[0]

            # Save the generated image
            output_image.save(output_path)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {base_filename}: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\nFinished generation. Processed {processed_count}/{len(files_to_process)} images saved to {output_dir}.")

if __name__ == "__main__":
    # --- Configuration --- 
    BASE_MODEL = "runwayml/stable-diffusion-v1-5"
    CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_seg" # ControlNet for Segmentation
    
    BASE_FACES_DIR = './data/synthetic_raw/base_faces'
    MASK_DIR = './data/masks/lesion_segmentation_maps'
    OUTPUT_SYNTHETIC_DIR = './data/synthetic_lesion'
    
    MAX_IMAGES_TO_PROCESS = 5 # Set to -1 to process all, or a small number for testing
    # Generation Parameters (tune these for desired results)
    PROMPT = "photo of a face, detailed skin texture, realistic lighting, high quality, 8k"
    NEG_PROMPT = "blurry, deformed, unrealistic, multiple faces, watermark, text, signature, cartoon, drawing, illustration, sketch, anime, duplicate, ugly, low quality, worst quality"
    STEPS = 30
    GUIDANCE = 7.5
    CONTROL_STRENGTH = 0.9 # Adjust strength of ControlNet guidance
    # --- End Configuration --- 

    # Load models
    pipeline, device = load_models(base_model_id=BASE_MODEL, controlnet_model_id=CONTROLNET_MODEL)

    if pipeline is not None:
        # Generate images
        generate_synthetic_images(
            pipe=pipeline,
            device=device,
            base_face_dir=BASE_FACES_DIR,
            mask_dir=MASK_DIR,
            output_dir=OUTPUT_SYNTHETIC_DIR,
            num_images=MAX_IMAGES_TO_PROCESS,
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            controlnet_conditioning_scale=CONTROL_STRENGTH
        )
    else:
        print("Could not load models. Exiting generation process.") 