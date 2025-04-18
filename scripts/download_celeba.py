import kagglehub
import os

# Define the target directory relative to the script location
# Moves up one level from 'scripts' to the project root, then into 'data/raw'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "celeba-dataset")

os.makedirs(TARGET_DATA_DIR, exist_ok=True)

print(f"Attempting to download CelebA dataset to: {TARGET_DATA_DIR}")

# Download latest version of the dataset directly to the target path
try:
    # Note: kagglehub.dataset_download might primarily use cache.
    # Let's try downloading to a specific path first, but be aware it might still use the cache primarily.
    # An alternative is to download to cache and then copy/move.
    path = kagglehub.dataset_download("google/celeba", path=TARGET_DATA_DIR)
    print(f"Kagglehub reported download path (might be cache): {path}")
    print(f"Please check the target directory for downloaded files: {TARGET_DATA_DIR}")
    # If the 'path' returned is not TARGET_DATA_DIR, files might be in the cache.
    # You might need to manually copy/move them from 'path' to TARGET_DATA_DIR if needed.

except Exception as e:
    print(f"An error occurred during download: {e}")
    print("Please ensure you have configured your Kaggle API credentials correctly.")
    print("See: https://github.com/Kaggle/kagglehub?tab=readme-ov-file#authenticate") 