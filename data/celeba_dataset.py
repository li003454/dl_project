import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CelebADataset(Dataset):
    """CelebA PyTorch Dataset."""

    def __init__(self, data_dir, partition, transform=None, attr_subset=None):
        """
        Args:
            data_dir (string): Path to the directory containing the CelebA dataset 
                                (expects 'archive' subdirectory with images and csv files).
            partition (string): Which dataset partition to load ('train', 'val', or 'test').
            transform (callable, optional): Optional transform to be applied on a sample.
            attr_subset (list, optional): List of attribute names to load. If None, loads all.
        """
        self.data_dir = os.path.join(data_dir, 'archive')
        # Adjusted path to account for nested directory structure observed
        self.img_dir = os.path.join(self.data_dir, 'img_align_celeba', 'img_align_celeba') 
        self.partition_file = os.path.join(self.data_dir, 'list_eval_partition.csv')
        self.attr_file = os.path.join(self.data_dir, 'list_attr_celeba.csv')
        self.transform = transform

        # Load partition data
        try:
            partitions_df = pd.read_csv(self.partition_file)
            self.attributes_df = pd.read_csv(self.attr_file)
        except FileNotFoundError as e:
            print(f"Error loading CSV file: {e}")
            print(f"Please ensure partition and attribute CSV files exist in {self.data_dir}")
            raise
            
        partition_map = {'train': 0, 'val': 1, 'test': 2}
        target_partition_id = partition_map.get(partition.lower())
        if target_partition_id is None:
            raise ValueError(f"Invalid partition specified: {partition}. Choose from train, val, test.")
        
        self.image_list = partitions_df[partitions_df['partition'] == target_partition_id]['image_id'].tolist()

        # Set image_id as index for faster lookup
        self.attributes_df.set_index('image_id', inplace=True)
        # Replace -1 with 0 for binary attributes
        self.attributes_df.replace(-1, 0, inplace=True)
        
        if attr_subset:
            # Filter attributes_df to only include the subset columns + original index
            try:
                self.attributes_df = self.attributes_df[attr_subset]
            except KeyError as e:
                print(f"Error selecting attributes: {e}. Available attributes are: {self.attributes_df.columns.tolist()}")
                raise

        # Filter the attributes dataframe to only include images from the target partition
        self.attributes_df = self.attributes_df.loc[self.image_list]
        
        print(f"Loaded {len(self.image_list)} images for partition: {partition}")
        if attr_subset:
             print(f"Selected attributes: {attr_subset}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
             print(f"Error: Image file not found at {img_path}")
             # Return a placeholder or handle the error as appropriate
             # For now, let's return None and handle it in the DataLoader or training loop
             return None, None # Or raise an exception

        attributes = self.attributes_df.loc[img_name].values
        attributes = torch.tensor(attributes, dtype=torch.float32)

        sample = {'image': image, 'attributes': attributes, 'image_id': img_name}

        if self.transform:
            # Apply transform only to the image
            sample['image'] = self.transform(sample['image'])

        return sample

# Example usage (demonstration purposes):
if __name__ == '__main__':
    # Define transformations (Resize and convert to Tensor)
    # You might want more complex transformations later (normalization, augmentation)
    img_size = 128
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Add normalization based on ImageNet stats or calculated from CelebA
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    # Path to the raw data directory (containing the 'archive' folder)
    raw_data_path = './data/raw/celeba-dataset' 

    # Create a dataset instance for the training partition
    try:
        train_dataset = CelebADataset(data_dir=raw_data_path, 
                                    partition='train', 
                                    transform=transform,
                                    attr_subset=['Male', 'Smiling']) # Example: Load only specific attributes

        # You can access samples like this:
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            
            # Add check for None before accessing sample elements
            if sample is not None and sample['image'] is not None:
                print("\nExample Sample:")
                print("Image Tensor Shape:", sample['image'].shape)
                print("Attributes Tensor:", sample['attributes'])
                print("Attribute Names:", ['Male', 'Smiling'])
                print("Image ID:", sample['image_id'])

                # Example of using DataLoader
                from torch.utils.data import DataLoader
                # Filter out None samples if any occurred during __getitem__
                # Check if the sample dictionary itself is valid and if the 'image' key is valid
                train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, 
                                        collate_fn=lambda batch: torch.utils.data.dataloader.default_collate([s for s in batch if s is not None and s.get('image') is not None]))
                
                try:
                    batch = next(iter(train_loader))
                    print("\nExample Batch:")
                    print("Image Batch Shape:", batch['image'].shape)
                    print("Attributes Batch Shape:", batch['attributes'].shape)
                    print("Image IDs in Batch:", batch['image_id'])
                except StopIteration:
                     print("\nCould not retrieve a batch. Dataset might be empty or all initial samples had errors.")

            else:
                print("\nCould not load the first sample correctly.")

        else:
             print("Training dataset is empty.")
             
    except ValueError as e:
        print(f"Error creating dataset: {e}")
    except FileNotFoundError as e:
         print(f"Error: A required file or directory was not found: {e}")
         print(f"Please ensure the CelebA dataset is correctly placed in {raw_data_path}/archive") 