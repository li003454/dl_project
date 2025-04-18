import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import os
import sys
from torchvision import transforms
import pandas as pd # Add pandas import needed by CelebADataset
from torch.utils.data import Dataset # Add Dataset import needed by CelebADataset

# <<< TEMPORARILY COPIED CelebADataset for testing FaceNetWrapper >>>
# In final code, use proper imports
class CelebADataset(Dataset):
    """CelebA PyTorch Dataset."""
    # ... (Full Class Definition as in data/celeba_dataset.py) ...
    # NOTE: Make sure the full class definition is copied here
    # Simplified version for brevity in this edit example:
    def __init__(self, data_dir, partition, transform=None, attr_subset=None):
        self.data_dir = os.path.join(data_dir, 'archive')
        self.img_dir = os.path.join(self.data_dir, 'img_align_celeba', 'img_align_celeba')
        self.partition_file = os.path.join(self.data_dir, 'list_eval_partition.csv')
        self.attr_file = os.path.join(self.data_dir, 'list_attr_celeba.csv')
        self.transform = transform
        try:
            partitions_df = pd.read_csv(self.partition_file)
            self.attributes_df = pd.read_csv(self.attr_file)
        except FileNotFoundError as e:
            print(f"Error loading CSV file: {e}")
            raise
        partition_map = {'train': 0, 'val': 1, 'test': 2}
        target_partition_id = partition_map.get(partition.lower())
        if target_partition_id is None: raise ValueError("Invalid partition")
        self.image_list = partitions_df[partitions_df['partition'] == target_partition_id]['image_id'].tolist()
        self.attributes_df.set_index('image_id', inplace=True)
        self.attributes_df.replace(-1, 0, inplace=True)
        if attr_subset: self.attributes_df = self.attributes_df[attr_subset]
        self.attributes_df = self.attributes_df.loc[self.image_list]
        print(f"Loaded {len(self.image_list)} images for partition: {partition}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            return None # Return None instead of tuple
        attributes = self.attributes_df.loc[img_name].values
        attributes = torch.tensor(attributes, dtype=torch.float32)
        sample = {'image': image, 'attributes': attributes, 'image_id': img_name}
        # Apply transform only if it exists and needed (example usage doesn't)
        # if self.transform: sample['image'] = self.transform(sample['image']) 
        return sample
# <<< END OF COPIED CelebADataset >>>


class FaceNetWrapper:
    """Wraps FaceNet (InceptionResnetV1) and MTCNN for face detection and embedding."""

    def __init__(self, device=None):
        """
        Initializes the MTCNN detector and InceptionResnetV1 embedder.

        Args:
            device (torch.device, optional): Device to run models on (e.g., 'cuda', 'cpu'). 
                                            Defaults to auto-detect CUDA or CPU.
        """
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"Using device: {self.device} for FaceNet models")

        # Initialize MTCNN for face detection
        # keep_all=True detects all faces, not just the one with highest probability
        # post_process=False returns raw tensors, needed for potential gradient flow
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709,
            keep_all=True, device=self.device, post_process=False
        )

        # Initialize InceptionResnetV1 for face embedding
        # Use VGGFace2 pre-trained weights for robust embeddings
        self.resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()

    def detect_and_extract_embeddings(self, img_pil):
        """
        Detects faces in a PIL image and extracts their embeddings.

        Args:
            img_pil (PIL.Image): Input image.

        Returns:
            tuple: (list of torch.Tensor, list of list): 
                   A list of face embeddings (tensors), and a list of bounding boxes [x1, y1, x2, y2].
                   Returns ([], []) if no faces are detected.
        """
        if img_pil is None:
            return [], []

        # Detect faces and get bounding boxes and probabilities
        # MTCNN expects batch dimension, add it and remove later
        # Use mtcnn.detect for boxes and probs, mtcnn() directly for aligned face tensors
        try:
            # Detect faces to get aligned face tensors directly
            # Note: mtcnn() applies post-processing (normalization) by default
            # We might need raw tensors if gradients are needed later. Let's use detect for now.
            
            # mtcnn.detect returns boxes, probs, landmarks
            boxes, probs, landmarks = self.mtcnn.detect(img_pil, landmarks=True)
            
            # Check if any faces were detected
            if boxes is None or len(boxes) == 0:
                return [], []

            # mtcnn() function directly returns aligned face tensors (requires PIL image)
            face_tensors = self.mtcnn(img_pil)
            if face_tensors is None:
                return [], []
            
            face_tensors = face_tensors.to(self.device)
            
            # Calculate embeddings for detected faces
            with torch.no_grad(): # Disable gradient calculation for standard inference
                embeddings = self.resnet(face_tensors)
                
            # Convert boxes to list of lists if needed
            boxes_list = boxes.tolist() if boxes is not None else []
            
            return embeddings, boxes_list
        except Exception as e:
            print(f"Error during face detection/embedding: {e}")
            # Might happen with very small images or unusual inputs
            return [], []

    def get_embeddings(self, face_tensor_batch):
        """
        Calculates embeddings for a batch of pre-aligned face tensors.

        Args:
            face_tensor_batch (torch.Tensor): Batch of face tensors (e.g., from MTCNN output), 
                                             expected shape (N, 3, 160, 160).

        Returns:
            torch.Tensor: Embeddings for the faces.
        """
        if face_tensor_batch is None or len(face_tensor_batch) == 0:
            return torch.empty((0, 512), device=self.device) # Return empty tensor with correct last dim
            
        face_tensor_batch = face_tensor_batch.to(self.device)
        
        # Calculate embeddings
        # Allow gradients if needed for attacks, otherwise use no_grad
        # For baseline integration, no_grad is fine. For attacks, this context might need removal.
        with torch.no_grad(): 
            embeddings = self.resnet(face_tensor_batch)
        return embeddings

    @staticmethod
    def embedding_distance(embedding1, embedding2):
        """
        Calculates the L2 distance (Euclidean distance) between two embeddings or batches of embeddings.
        
        Args:
            embedding1 (torch.Tensor): First embedding or batch (N, D).
            embedding2 (torch.Tensor): Second embedding or batch (N, D) or single (1, D).
        
        Returns:
            torch.Tensor: Distance(s).
        """
        if embedding1.ndim == 1:
            embedding1 = embedding1.unsqueeze(0)
        if embedding2.ndim == 1:
            embedding2 = embedding2.unsqueeze(0)
        
        # Calculate squared Euclidean distance
        dist_sq = torch.sum(torch.pow(embedding1 - embedding2, 2), dim=1)
        # Return Euclidean distance
        return torch.sqrt(dist_sq)

# Example usage:
if __name__ == '__main__':
    # Assuming you are in the project root directory
    # Adjust path if running from elsewhere
    # No longer need sys.path modification as Dataset is defined locally
    # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # sys.path.append(project_root)
    # from data.celeba_dataset import CelebADataset # Import is no longer needed
    
    # Use a standard transform suitable for FaceNet (usually minimal)
    transform = transforms.Compose([
        # MTCNN handles resizing internally, but maybe a general resize first
        # transforms.Resize((256, 256)) 
        #ToTensor() is usually handled by MTCNN when getting faces
    ])

    # Load a sample image using the CelebA dataset loader
    # Assume script is run from project root for this relative path to work
    celeba_data_dir = os.path.join('.', 'data', 'raw', 'celeba-dataset') 
    try:
        # Load test partition as it's smaller
        test_dataset = CelebADataset(data_dir=celeba_data_dir, partition='test', transform=None) # Don't transform yet
        
        if len(test_dataset) > 0:
            # Get a sample PIL image
            # Add a loop to find the first valid sample, in case the first image is missing
            sample_data = None
            for i in range(len(test_dataset)):
                sample_data = test_dataset[i]
                if sample_data is not None and sample_data.get('image') is not None:
                     print(f"Using sample index: {i}")
                     break # Found a valid sample
            
            if sample_data is not None and sample_data.get('image') is not None:
                img_pil = sample_data['image']
                print(f"Loaded sample image ID: {sample_data['image_id']}")
                
                # Initialize the wrapper
                facenet = FaceNetWrapper()

                # Detect faces and extract embeddings
                embeddings, boxes = facenet.detect_and_extract_embeddings(img_pil)

                if embeddings is not None and len(embeddings) > 0:
                    print(f"Detected {len(boxes)} face(s).")
                    print(f"Extracted {len(embeddings)} embedding(s) with shape: {embeddings[0].shape}")
                    
                    # Example: Compare first embedding to itself (distance should be 0)
                    dist = FaceNetWrapper.embedding_distance(embeddings[0], embeddings[0])
                    print(f"Distance between first embedding and itself: {dist.item()}")
                    
                    if len(embeddings) > 1:
                       dist_diff = FaceNetWrapper.embedding_distance(embeddings[0], embeddings[1])
                       print(f"Distance between first two embeddings: {dist_diff.item()}")
                else:
                    print("No faces detected in the sample image.")
            else:
                print("Could not load a valid sample from the dataset (tried multiple indices). Maybe check image paths or file integrity?")
        else:
            print("Test dataset is empty or could not be loaded.")
            
    except Exception as e:
        print(f"An error occurred during FaceNet wrapper example: {e}")
        import traceback
        traceback.print_exc()
        print("Ensure the CelebA dataset is correctly downloaded and paths are correct.")
        print("You might need to install dataset dependencies if not done yet.") 