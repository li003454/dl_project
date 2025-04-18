import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import os
import sys
# Need these for the example usage part
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

# <<< TEMPORARILY COPIED CelebADataset for testing wrapper >>>
# In final code, use proper imports
class CelebADataset(Dataset): # Simplified version for testing
    def __init__(self, data_dir, partition, transform=None, attr_subset=None):
        self.data_dir = os.path.join(data_dir, 'archive')
        self.img_dir = os.path.join(self.data_dir, 'img_align_celeba', 'img_align_celeba')
        self.partition_file = os.path.join(self.data_dir, 'list_eval_partition.csv')
        try:
            partitions_df = pd.read_csv(self.partition_file)
        except FileNotFoundError as e:
            print(f"Error loading CSV file: {e}"); raise
        partition_map = {'train': 0, 'val': 1, 'test': 2}
        target_partition_id = partition_map.get(partition.lower())
        if target_partition_id is None: raise ValueError("Invalid partition")
        self.image_list = partitions_df[partitions_df['partition'] == target_partition_id]['image_id'].tolist()
        print(f"Loaded {len(self.image_list)} images for partition: {partition}")
    def __len__(self): return len(self.image_list)
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try: image = Image.open(img_path).convert('RGB')
        except FileNotFoundError: print(f"Error: Image file not found at {img_path}"); return None
        return {'image': image, 'image_id': img_name}
# <<< END OF COPIED CelebADataset >>>

class InstructPix2PixWrapper:
    """Wraps the InstructPix2Pix diffusion model for image editing."""

    def __init__(self, model_id="timbrooks/instruct-pix2pix", device=None):
        """
        Initializes the InstructPix2Pix pipeline.

        Args:
            model_id (str): The Hugging Face model ID for InstructPix2Pix.
            device (torch.device, optional): Device to run the model on. Defaults to auto.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device} for InstructPix2Pix")

        # Load the pipeline
        # Using float16 for memory efficiency on GPU
        try:
            self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32, 
                safety_checker=None # Disable safety checker for potentially less restricted editing
            )
            self.pipe.to(self.device)
            # Optional: Use a different scheduler if desired
            # self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
            print(f"Loaded InstructPix2Pix model: {model_id}")
        except Exception as e:
             print(f"Error loading InstructPix2Pix model: {e}")
             print("Ensure you have an internet connection and enough memory.")
             self.pipe = None

    def edit_image(self, image_pil, instruction, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7):
        """
        Applies an editing instruction to a PIL image.

        Args:
            image_pil (PIL.Image): The input image to edit.
            instruction (str): The text instruction describing the edit.
            num_inference_steps (int): Number of denoising steps.
            image_guidance_scale (float): Strength of image guidance.
            guidance_scale (float): Strength of text guidance.

        Returns:
            PIL.Image or None: The edited image, or None if an error occurred.
        """
        if self.pipe is None or image_pil is None:
            print("Error: Pipeline not loaded or input image is invalid.")
            return None
            
        # Ensure image is RGB
        image_pil = image_pil.convert("RGB")
            
        try:
            # Run inference
            edited_images = self.pipe(
                instruction, 
                image=image_pil,
                num_inference_steps=num_inference_steps, 
                image_guidance_scale=image_guidance_scale, 
                guidance_scale=guidance_scale
            ).images
            
            # The pipeline returns a list of images, typically just one
            if edited_images and len(edited_images) > 0:
                return edited_images[0]
            else:
                 print("Error: Editing pipeline did not return an image.")
                 return None
        except Exception as e:
            print(f"Error during image editing: {e}")
            return None

# Example usage:
if __name__ == '__main__':
    # Assume script is run from project root
    celeba_data_dir = os.path.join('.', 'data', 'raw', 'celeba-dataset') 
    output_dir = os.path.join('.', 'output', 'instructpix2pix_test')
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load a sample image (using the temporary Dataset definition)
        test_dataset = CelebADataset(data_dir=celeba_data_dir, partition='test')
        
        sample_data = None
        if len(test_dataset) > 0:
             for i in range(5): # Try first 5 samples
                 sample_data = test_dataset[i]
                 if sample_data is not None and sample_data.get('image') is not None:
                     print(f"Using sample index: {i}")
                     break
        
        if sample_data is not None and sample_data.get('image') is not None:
            original_image = sample_data['image']
            image_id = sample_data['image_id']
            print(f"Loaded sample image ID: {image_id}")

            # Initialize the wrapper
            editor = InstructPix2PixWrapper()

            if editor.pipe is not None:
                # Define an editing instruction
                instruction = "make the person wear glasses" # Example instruction
                print(f"Applying instruction: '{instruction}'")

                # Perform the edit
                edited_image = editor.edit_image(original_image, instruction)

                if edited_image:
                    print("Editing successful.")
                    # Save or display the images
                    original_image.save(os.path.join(output_dir, f"{image_id}_original.png"))
                    edited_image.save(os.path.join(output_dir, f"{image_id}_edited_{instruction.replace(' ','_')}.png"))
                    print(f"Saved original and edited images to: {output_dir}")

                    # Optional: Display images using matplotlib
                    # try:
                    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    #     axes[0].imshow(original_image)
                    #     axes[0].set_title("Original")
                    #     axes[0].axis('off')
                    #     axes[1].imshow(edited_image)
                    #     axes[1].set_title(f"Edited: {instruction}")
                    #     axes[1].axis('off')
                    #     plt.tight_layout()
                    #     plt.show()
                    # except NameError:
                    #      print("Matplotlib not available or display unavailable. Skipping plot.")

                else:
                    print("Editing failed.")
            else:
                 print("Skipping edit example because model could not be loaded.")
        else:
             print("Could not load a valid sample from the dataset.")

    except Exception as e:
        print(f"An error occurred during InstructPix2Pix wrapper example: {e}")
        import traceback
        traceback.print_exc() 