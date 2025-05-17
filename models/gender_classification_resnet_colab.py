### Gender Recognition ResNet Model - Colab Version
### This script is adapted for Google Colab - training a gender recognition model on 512x512 images using GPU
### You must have the required dataset to train the model and you may also need to change the paths.

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/'CSCI5527 deep learning'/'Final_Project'/'Final_Project_Final'




# access the dataset(you may need to change your own path here)
train_dir = 'CelebA_HQ_face_gender_dataset/train'
test_dir = 'CelebA_HQ_face_gender_dataset_test_1000'



# Confirm if folders exist
print(f"Train directory exists: {os.path.exists(train_dir)}")
print(f"Test directory exists: {os.path.exists(test_dir)}")

# Check number of samples in train and test sets
if os.path.exists(train_dir) and os.path.exists(test_dir):
    male_train_count = len(os.listdir(os.path.join(train_dir, 'male')))
    female_train_count = len(os.listdir(os.path.join(train_dir, 'female')))

    male_test_count = len(os.listdir(os.path.join(test_dir, 'male')))
    female_test_count = len(os.listdir(os.path.join(test_dir, 'female')))

    print(f"Training set: Males = {male_train_count}, Females = {female_train_count}, Total = {male_train_count + female_train_count}")
    print(f"Test set: Males = {male_test_count}, Females = {female_test_count}, Total = {male_test_count + female_test_count}")



# 4. Define data preprocessing
# 4. Define data preprocessing
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # Scales images to [0, 1]
    transforms.Lambda(lambda x: x * 2.0 - 1.0) # Rescales [0,1] to [-1,1]
])

test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(), # Scales images to [0, 1]
    transforms.Lambda(lambda x: x * 2.0 - 1.0) # Rescales [0,1] to [-1,1]
])

# 5. Load datasets
# Ensure the paths are correct and datasets are available before running this
if os.path.exists(train_dir) and os.path.exists(test_dir): # Added check
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Adjust batch size based on A100 GPU (40GB memory) or other available hardware
    # batch_size = 64  # For A100 GPU, increase batch size for training efficiency
    # num_workers = 4  # Increase worker threads for faster data loading
    # Define batch_size and num_workers if not defined globally or if you want to override
    if 'batch_size' not in globals(): batch_size = 32 # Default if not set
    if 'num_workers' not in globals(): num_workers = 2 # Default if not set


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Display class information
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
else:
    print("Error: Train or Test directory not found. Please check paths.")
    # Exit or handle the error appropriately
    # For now, we'll create dummy loaders if paths are not found to prevent further errors in a notebook setting
    # but in a real script, you'd likely exit.
    class_names = ['female', 'male'] # Dummy
    train_loader = None
    test_loader = None


# Function to display Tensor images
def imshow(inp, title=None):
    """Display Tensor image (normalized to [-1, 1])"""
    inp = inp.numpy().transpose((1, 2, 0)) # Convert from Tensor image CHW to numpy HWC
    # De-normalize from [-1, 1] to [0, 1]
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean # Formula: original = normalized * std + mean
    inp = np.clip(inp, 0, 1) # Clip values to [0,1] range for display
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


# Function to get and display some preprocessed samples
def show_processed_samples(dataloader, class_names_list, num_samples=8): # Changed class_names to class_names_list to avoid conflict
    if dataloader is None: # Check if dataloader is valid
        print("Dataloader not initialized, cannot show samples.")
        return
    # Get a batch of data
    images, labels = next(iter(dataloader))

    # Create image grid
    fig = plt.figure(figsize=(15, 8))
    rows = num_samples // 4 + (1 if num_samples % 4 else 0)

    for i in range(min(num_samples, len(images))):
        ax = plt.subplot(rows, 4, i + 1)
        ax.axis('off')
        ax.set_title(f'Class: {class_names_list[labels[i]]}') # Use class_names_list

        # De-normalize image for visualization
        imshow(images[i])

    plt.tight_layout()
    plt.savefig('processed_samples.png')
    plt.show()
    print("Processed samples displayed and saved to 'processed_samples.png'")

# Display preprocessed samples (only if train_loader is valid)
print("\nDisplaying preprocessed training samples...")
if train_loader:
    show_processed_samples(train_loader, class_names)



# 7. Define the model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
model.eval()

# Modify the final layer for binary classification
num_ftrs = model.fc.in_features # Get number of features from the penultimate layer
model.fc = nn.Linear(num_ftrs, len(class_names)) # Replace fc layer with a new one for 2 classes
model = model.to(device) # Move model to the designated device

# 8. Define loss function and optimizer
criterion = nn.CrossEntropyLoss() # Suitable for multi-class (and binary) classification with logits output
# Use a larger learning rate to accommodate a larger batch size
optimizer = optim.Adam(model.parameters(), lr=0.0003)  # Slightly increased learning rate for batch size 64 (adjust if bs differs)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True) # Reduce LR when a metric has stopped improving

# 9. Define training function
def train_model(model_to_train, loss_criterion, opt, lr_scheduler, num_epochs=10): # Renamed some params to avoid conflict
    since = time.time()

    best_model_wts_train = model_to_train.state_dict() # Use model_to_train
    best_acc_train = 0.0 # Use best_acc_train

    # Store training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    if train_loader is None or test_loader is None: # Check if loaders are valid
        print("Dataloaders not initialized. Cannot train model.")
        return model_to_train, history


    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model_to_train.train() # Set model to training mode
        running_loss_train = 0.0
        running_corrects_train = 0

        for inputs, labels in train_loader: # Iterate over training data
            inputs = inputs.to(device)
            labels = labels.to(device)

            opt.zero_grad() # Zero the parameter gradients

            outputs = model_to_train(inputs) # Forward pass
            _, preds = torch.max(outputs, 1) # Get predictions
            loss = loss_criterion(outputs, labels) # Calculate loss

            loss.backward() # Backward pass
            opt.step() # Optimize

            running_loss_train += loss.item() * inputs.size(0)
            running_corrects_train += torch.sum(preds == labels.data)

        epoch_loss_train = running_loss_train / len(train_dataset)
        epoch_acc_train = running_corrects_train.double() / len(train_dataset)

        history['train_loss'].append(epoch_loss_train)
        history['train_acc'].append(epoch_acc_train.item())

        print(f'Train Loss: {epoch_loss_train:.4f} Acc: {epoch_acc_train:.4f}')

        # Testing/Validation phase
        model_to_train.eval() # Set model to evaluation mode
        running_loss_test = 0.0
        running_corrects_test = 0

        with torch.no_grad(): # Disable gradient calculation
            for inputs, labels in test_loader: # Iterate over test data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model_to_train(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_criterion(outputs, labels)

                running_loss_test += loss.item() * inputs.size(0)
                running_corrects_test += torch.sum(preds == labels.data)

        epoch_loss_test = running_loss_test / len(test_dataset)
        epoch_acc_test = running_corrects_test.double() / len(test_dataset)

        history['test_loss'].append(epoch_loss_test)
        history['test_acc'].append(epoch_acc_test.item())

        print(f'Test Loss: {epoch_loss_test:.4f} Acc: {epoch_acc_test:.4f}')

        # Adjust learning rate based on validation accuracy
        lr_scheduler.step(epoch_acc_test)

        # Save the best model weights
        if epoch_acc_test > best_acc_train:
            best_acc_train = epoch_acc_test
            best_model_wts_train = model_to_train.state_dict()

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best test Acc: {best_acc_train:.4f}')

    # Load best model weights
    model_to_train.load_state_dict(best_model_wts_train)
    return model_to_train, history

# 10. Model evaluation function
def evaluate_model(model_to_eval, data_loader): # Renamed params
    if data_loader is None:
        print("Dataloader not initialized. Cannot evaluate model.")
        return 0.0, [], []

    model_to_eval.eval() # Set model to evaluation mode
    correct_eval = 0
    total_eval = 0
    all_preds_eval = []
    all_labels_eval = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_to_eval(inputs)
            _, preds = torch.max(outputs, 1)

            total_eval += labels.size(0)
            correct_eval += (preds == labels).sum().item()

            all_preds_eval.extend(preds.cpu().numpy())
            all_labels_eval.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy_eval = 100 * correct_eval / total_eval if total_eval > 0 else 0
    print(f'Test Accuracy: {accuracy_eval:.2f}%')

    # Print accuracy for each class
    for i, class_name_item in enumerate(class_names): # Use different var name for class_name
        class_correct_eval = np.sum((np.array(all_preds_eval) == i) & (np.array(all_labels_eval) == i))
        class_total_eval = np.sum(np.array(all_labels_eval) == i)
        if class_total_eval > 0:
            print(f'Accuracy of {class_name_item}: {100 * class_correct_eval / class_total_eval:.2f}%')

    return accuracy_eval, all_preds_eval, all_labels_eval

# 11. Visualize some test results
def visualize_results(model_to_viz, data_loader, class_names_list, num_images=8): # Renamed params
    if data_loader is None:
        print("Dataloader not initialized. Cannot visualize results.")
        return

    was_training = model_to_viz.training
    model_to_viz.eval() # Set model to evaluation mode

    images_so_far = 0
    # Calculate required number of rows
    rows = (num_images + 3) // 4  # Assuming 4 columns, changed for better layout if num_images is not multiple of 4
    if rows == 0 : rows = 1 # ensure at least one row
    cols = 4 # Define number of columns for subplot
    if num_images < cols: cols = num_images # adjust if fewer images than cols
    fig = plt.figure(figsize=(15, 3 * rows)) # Adjusted figure size

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_to_viz(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                # Ensure subplot index does not exceed rows * cols
                if images_so_far <= rows * cols:
                    ax = plt.subplot(rows, cols, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'Pred: {class_names_list[preds[j]]}\nTrue: {class_names_list[labels[j]]}') # More compact title

                    # De-normalize image for visualization
                    inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                    mean_viz = np.array([0.485, 0.456, 0.406]) # Use different var name
                    std_viz = np.array([0.229, 0.224, 0.225])  # Use different var name
                    inp = std_viz * inp + mean_viz
                    inp = np.clip(inp, 0, 1)

                    ax.imshow(inp)

                if images_so_far == num_images:
                    model_to_viz.train(mode=was_training) # Restore original training mode
                    plt.tight_layout()
                    plt.savefig('sample_predictions.png')
                    plt.show()
                    return

    model_to_viz.train(mode=was_training) # Restore original training mode
    # If loop finishes before num_images, ensure processed images are shown
    if images_so_far > 0:
        plt.tight_layout()
        plt.savefig('sample_predictions.png')
        plt.show()

# 12. Plot training result curves
def plot_training_results(train_history): # Renamed param
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_history['train_loss'], label='Train Loss')
    plt.plot(train_history['test_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_history['train_acc'], label='Train Accuracy')
    plt.plot(train_history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Test Accuracy')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# 13. Start model training
num_epochs_train = 15  # Increased training epochs for a larger dataset (e.g., 30k images) to ensure sufficient training
# Check if train_loader is valid before starting training
if train_loader and test_loader:
    trained_model, history_data = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs_train) # Renamed vars

    # 14. Save the model
    torch.save(trained_model.state_dict(), 'gender_resnet50_model.pth')
    print("Model saved successfully!")

    # If you need to save to Google Drive, uncomment the line below
    # torch.save(trained_model.state_dict(), '/content/drive/MyDrive/gender_resnet50_model.pth')

    # 15. Plot training results
    plot_training_results(history_data) # Use renamed var

    # 16. Final evaluation
    final_accuracy, all_final_preds, all_final_labels = evaluate_model(trained_model, test_loader) # Renamed vars

    # 17. Visualize results
    visualize_results(trained_model, test_loader, class_names)
else:
    print("Skipping training and evaluation as dataloaders are not initialized.")