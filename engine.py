import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
from ..models.VAE_CNN import VariationalAutoEncoder
import numpy as np
import pandas as pd

###################################### ADDING NOISE ######################################
def add_noise(x, noise_factor=0.3):
    noise = torch.randn_like(x) * noise_factor
    noisy = x + noise  # This preserves requires_grad from x
    return torch.clamp(noisy, 0., 1.)

###################################### TRAINING ######################################
def train_model(
    model,
    train_loader,
    optimizer,
    loss_fn,
    device,
    num_epochs,
    input_dim,
    add_noise_flag=False,
    noise_factor=0.3,
    save_path="vae_model.pth",
    cnn_flag=False,
    noisy_classes=None  # New parameter
):
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        epoch_loss = 0

        if not cnn_flag:
            for i, (x, y) in loop:  # Now using (x, y) instead of just (x, _)
                x = x.to(device).view(x.shape[0], input_dim)

                if add_noise_flag:
                    if i == 0 and epoch == 0:
                        print(f"Running DVAE for classes {noisy_classes}")
                    x = x.requires_grad_(True)
                    
                    if noisy_classes is not None:
                        # Create mask for samples that should be noisy
                        noise_mask = torch.tensor([label in noisy_classes for label in y]).to(device)
                        noise_mask = noise_mask.view(-1, 1).expand_as(x)
                        
                        # Add noise only to selected classes
                        x_noisy = x.clone()
                        noise = torch.randn_like(x) * noise_factor
                        x_noisy[noise_mask] = torch.clamp(x[noise_mask] + noise[noise_mask], 0., 1.)
                        if i == 0 and epoch == 0:
                            print(f"x_noisy requires_grad: {x_noisy.requires_grad}")
                    else:
                        # Add noise to all samples if no specific classes specified
                        x_noisy = add_noise(x, noise_factor)
                    x_input = x_noisy
                else:
                    if i == 0 and epoch == 0:
                        print("Running Normal VAE")
                    x_input = x

                # Forward pass
                x_reconstructed, mu, logvar = model(x_input)

                # Loss calculation
            # ADDING KL WEIGHT TO THIS MAYBE MAKE A HYPERPARAMETER FOR THIS
                kl_weight = 1
                reconstruction_loss = loss_fn(x_reconstructed, x)  # Compare to original clean x
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + kl_weight * kl_div

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())

        else:
            
            # CNN training path
            for i, (x, y) in loop:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")

    torch.save(model.state_dict(), save_path)


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