import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
from ..models.VAE_CNN import VariationalAutoEncoder
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os


###################################### INFERENCE ######################################
# Define inference function
def inference(model, test_loader, device, num_samples=8, 
              save_path="inference_reconstruction.png", dataset_save_path="generated_dataset.pt",
              cnn_flag=False):
    if cnn_flag:
        # For CNN, we don't need to save reconstructions
        print("CNN model - skipping reconstruction visualization")
        return
    
    # Get one batch and limit to num_samples
    test_batch = next(iter(test_loader))[0][:num_samples].to(device)  # Only take num_samples images
    test_batch_flat = test_batch.view(test_batch.size(0), -1)
    
    # Inference
    with torch.no_grad():
        recon_batch, _, _ = model(test_batch_flat)
    
    # Reshape for visualization
    recon_batch = recon_batch.view(-1, 1, 28, 28)
    original = test_batch.view(-1, 1, 28, 28)
    
    # Concatenate and save
    comparison = torch.cat([original, recon_batch])
    save_image(comparison, save_path, nrow=num_samples)  # Changed to num_samples*2 to show pairs in rows
    
    # Store generated images in a new dataset
    generated_dataset = TensorDataset(recon_batch)
    torch.save(generated_dataset, dataset_save_path)
    print(f"Generated dataset saved to {dataset_save_path}")

######################################################## EVALUATE CNN ###############################################################

def evaluate_cnn(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10  # For each class
    class_total = [0] * 10    # For each class
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, predicted = torch.max(y_pred, 1)
            
            # Update total and correct counts
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            # Update per-class counts
            for i in range(y.size(0)):
                label = y[i]
                pred = predicted[i]
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
    
    # Calculate overall accuracy
    accuracy = correct / total
    
    # Print overall accuracy
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    # Create results dictionary for DataFrame
    results = {
        'Class': list(range(10)),
        'Correct': class_correct,
        'Total': class_total,
        'Accuracy': [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)]
    }
    
    # Add overall accuracy to results
    results['Overall_Accuracy'] = accuracy
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for i in range(10):
        if class_total[i] > 0:
            class_accuracy = class_correct[i] / class_total[i]
            print(f"Class {i}: {class_accuracy:.2%} ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"Class {i}: No samples")
    
    return results_df

############################### UTILS #######################################
def plot_noise_comparison(all_losses, save_path='Results/loss_comparison.png'):
    """Plot loss curves for different noise types"""
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create plot
    for noise_type in all_losses['noise_types'].unique():
        data = all_losses[all_losses['noise_types'] == noise_type]
        plt.plot(data['epoch'], data['loss'], label=noise_type, marker='o', markersize=3)
    
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Loss Comparison Across Different Noise Types')
    plt.legend(title='Noise Types')
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()