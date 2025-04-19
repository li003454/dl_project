import os
import sys
from PIL import Image
from torchvision import transforms
from tqdm import tqdm # For progress bar

# Adjust path to import CelebADataset from the data directory
# Assuming this script is run from the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root) 

# Temporarily copy CelebADataset class definition here to avoid import issues during dev
# In final structure, ensure proper package setup or PYTHONPATH
import pandas as pd
from torch.utils.data import Dataset
class CelebADataset(Dataset): # Simplified version from data/celeba_dataset.py
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
        # Only need image list for this script

    def __len__(self): return len(self.image_list)
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try: image = Image.open(img_path).convert('RGB')
        except FileNotFoundError: print(f"Warning: Image file not found at {img_path}"); return None
        # Return only image and name for this script
        return {'image': image, 'image_id': img_name}
# --- End of temporary CelebADataset ---

def prepare_base_faces(celeba_root_dir, output_dir, num_faces=200, target_size=512, partition='test'):
    """
    Selects images from CelebA, resizes them, and saves them as base faces.

    Args:
        celeba_root_dir (str): Path to the root CelebA directory (containing 'archive').
        output_dir (str): Directory to save the prepared base face images.
        num_faces (int): Number of faces to prepare.
        target_size (int): The target square size for the output images.
        partition (str): Which CelebA partition to use ('train', 'val', 'test').
    """
    print(f"Preparing {num_faces} base faces from CelebA {partition} partition...")
    print(f"Source directory: {celeba_root_dir}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)

    # Define the transformation: Resize
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
    ])

    # Load the CelebA dataset (using the temporary definition above)
    try:
        dataset = CelebADataset(data_dir=celeba_root_dir, partition=partition)
    except Exception as e:
        print(f"Error loading CelebA dataset: {e}")
        print("Please ensure the dataset is correctly downloaded and paths are correct.")
        return

    count = 0
    processed_ids = []
    # Use tqdm for progress bar
    pbar = tqdm(total=num_faces, desc="Processing faces")
    for i in range(len(dataset)):
        if count >= num_faces:
            break
            
        sample = dataset[i]
        if sample is None or sample.get('image') is None:
            print(f"Skipping index {i} due to loading error.")
            continue

        img_pil = sample['image']
        img_id = sample['image_id']
        
        # Apply resize transformation
        try:
            resized_img = transform(img_pil)
        except Exception as e:
             print(f"Error transforming image {img_id}: {e}")
             continue # Skip if transform fails

        # Save the processed image
        output_filename = os.path.join(output_dir, img_id)
        try:
            resized_img.save(output_filename)
            processed_ids.append(img_id)
            count += 1
            pbar.update(1)
        except Exception as e:
            print(f"Error saving image {output_filename}: {e}")
            
    pbar.close()
    print(f"\nSuccessfully prepared and saved {count} base face images to {output_dir}.")
    
    # Optional: Save the list of processed image IDs for reference
    ids_list_path = os.path.join(output_dir, '_prepared_ids.txt')
    try:
        with open(ids_list_path, 'w') as f:
            for img_id in processed_ids:
                f.write(f"{img_id}\n")
        print(f"List of prepared image IDs saved to {ids_list_path}")
    except Exception as e:
        print(f"Error saving list of processed IDs: {e}")

if __name__ == "__main__":
    # Configuration
    CELEBA_DATA_ROOT = './data/raw/celeba-dataset'  # Path relative to project root
    OUTPUT_BASE_FACES_DIR = './data/synthetic_raw/base_faces'
    NUM_FACES_TO_PREPARE = 200 # Adjust as needed
    TARGET_IMAGE_SIZE = 512
    DATA_PARTITION = 'test' # Use test set for smaller initial run

    prepare_base_faces(
        celeba_root_dir=CELEBA_DATA_ROOT,
        output_dir=OUTPUT_BASE_FACES_DIR,
        num_faces=NUM_FACES_TO_PREPARE,
        target_size=TARGET_IMAGE_SIZE,
        partition=DATA_PARTITION
    ) 