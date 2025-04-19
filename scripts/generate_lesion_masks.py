import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from PIL import Image # To read base image size if needed, though we assume 512x512

def define_lesion_params():
    """Defines lesion types, colors, shapes, placement zones, and size ranges."""
    # Define colors (BGR format for OpenCV)
    colors = {
        'background': (0, 0, 0), # Black
        'mole': (0, 64, 128),     # Dark Brown
        'patch': (0, 0, 255),    # Red
        'pigment': (128, 128, 128) # Grey
    }
    lesion_types = list(colors.keys())[1:] # Exclude background

    # Define placement zones as relative coordinates (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    # Based on a 512x512 image. Avoid center (nose/mouth) and eyes.
    zones = {
        'forehead': (0.2, 0.1, 0.8, 0.3),  # Top portion
        'left_cheek': (0.1, 0.35, 0.4, 0.7),
        'right_cheek': (0.6, 0.35, 0.9, 0.7),
        'chin': (0.3, 0.75, 0.7, 0.9)
    }

    # Define size ranges as relative to image width/height
    size_ranges = {
        'mole': (0.02, 0.05), # Relatively small
        'patch': (0.05, 0.15),# Larger
        'pigment': (0.04, 0.12) # Medium
    }
    
    # Define shapes
    shapes = ['circle', 'ellipse']

    return colors, lesion_types, zones, size_ranges, shapes

def generate_segmentation_mask(output_path, base_image_size=(512, 512)):
    """
    Generates a single segmentation mask with a randomly placed lesion.
    Args:
        output_path (str): Path to save the generated mask.
        base_image_size (tuple): The (width, height) of the base image.
    Returns:
        bool: True if mask was generated successfully, False otherwise.
    """
    colors, lesion_types, zones, size_ranges, shapes = define_lesion_params()
    width, height = base_image_size

    # Create a black background
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    mask[:] = colors['background']

    # Choose random elements
    lesion_type = random.choice(lesion_types)
    zone_key = random.choice(list(zones.keys()))
    shape = random.choice(shapes)
    lesion_color = colors[lesion_type]
    min_rel_size, max_rel_size = size_ranges[lesion_type]

    # Calculate placement zone in absolute coordinates
    zone_coords = zones[zone_key]
    x1_zone = int(zone_coords[0] * width)
    y1_zone = int(zone_coords[1] * height)
    x2_zone = int(zone_coords[2] * width)
    y2_zone = int(zone_coords[3] * height)

    # Calculate random size (radius or axes)
    size_factor = random.uniform(min_rel_size, max_rel_size)
    major_axis = int(size_factor * width / 2) # Use width for consistency
    minor_axis = major_axis # Default for circle
    if shape == 'ellipse':
        minor_axis = int(major_axis * random.uniform(0.5, 0.9)) # Make it elliptical

    # Calculate random center within the zone, ensuring shape fits
    # Add padding based on major axis to avoid drawing outside zone boundaries
    try:
        center_x = random.randint(x1_zone + major_axis, x2_zone - major_axis)
        center_y = random.randint(y1_zone + major_axis, y2_zone - major_axis)
    except ValueError: # Zone might be too small for the chosen size
        print(f"Warning: Zone '{zone_key}' might be too small for lesion size {major_axis}. Adjusting center calculation.")
        # Fallback: place closer to the middle of the zone if possible
        cx_min = max(x1_zone, x1_zone + major_axis)
        cx_max = min(x2_zone, x2_zone - major_axis)
        cy_min = max(y1_zone, y1_zone + major_axis)
        cy_max = min(y2_zone, y2_zone - major_axis)
        if cx_max <= cx_min or cy_max <= cy_min:
             print(f"Error: Cannot place lesion of size {major_axis} in zone '{zone_key}'. Skipping mask generation for {output_path}")
             return False # Cannot place lesion
        center_x = random.randint(cx_min, cx_max)
        center_y = random.randint(cy_min, cy_max)

    # Draw the shape
    center = (center_x, center_y)
    axes = (major_axis, minor_axis)
    
    if shape == 'circle':
        cv2.circle(mask, center, major_axis, lesion_color, -1) # -1 thickness fills the circle
    elif shape == 'ellipse':
        angle = random.uniform(0, 180) # Random orientation
        cv2.ellipse(mask, center, axes, angle, 0, 360, lesion_color, -1)

    # Save the mask
    try:
        cv2.imwrite(output_path, mask)
        return True
    except Exception as e:
        print(f"Error saving mask {output_path}: {e}")
        return False

def batch_generate_masks(base_face_dir, mask_output_dir, base_image_size=(512, 512)):
    """
    Generates segmentation masks for all base faces found in a directory.

    Args:
        base_face_dir (str): Directory containing the base face images.
        mask_output_dir (str): Directory to save the generated masks.
        base_image_size (tuple): Expected size of base images (width, height).
    """
    print(f"Generating segmentation masks for base faces in: {base_face_dir}")
    print(f"Output directory for masks: {mask_output_dir}")
    
    os.makedirs(mask_output_dir, exist_ok=True)

    try:
        # Prefer using the saved list if available, otherwise list directory
        ids_list_path = os.path.join(base_face_dir, '_prepared_ids.txt')
        if os.path.exists(ids_list_path):
             with open(ids_list_path, 'r') as f:
                 base_face_files = [line.strip() for line in f if line.strip()]
             print(f"Found list of {len(base_face_files)} prepared IDs.")
        else:
            print("Warning: '_prepared_ids.txt' not found. Listing directory instead.")
            base_face_files = [f for f in os.listdir(base_face_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not base_face_files:
                 print(f"Error: No base face images found in {base_face_dir}")
                 return
            print(f"Found {len(base_face_files)} image files in directory.")
            
    except Exception as e:
        print(f"Error accessing base face directory or ID list: {e}")
        return

    generated_count = 0
    for filename in tqdm(base_face_files, desc="Generating masks"):
        output_filename = os.path.join(mask_output_dir, filename + ".png") # Save mask as png
        if generate_segmentation_mask(output_filename, base_image_size):
             generated_count += 1
             
    print(f"\nFinished generating masks. Successfully created {generated_count} masks out of {len(base_face_files)} base faces.")

if __name__ == "__main__":
    # Configuration
    BASE_FACES_INPUT_DIR = './data/synthetic_raw/base_faces' # Directory created by prepare_base_faces.py
    MASK_OUTPUT_DIR = './data/masks/lesion_segmentation_maps'
    IMAGE_SIZE = (512, 512) # Should match the size used in prepare_base_faces.py

    batch_generate_masks(
        base_face_dir=BASE_FACES_INPUT_DIR,
        mask_output_dir=MASK_OUTPUT_DIR,
        base_image_size=IMAGE_SIZE
    ) 