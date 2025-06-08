import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from ..models.VAE_CNN import VQVAE, VectorQuantizer
from ..engine.engine import train_model, inference
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os

########################### RUN THE VQVAE #########################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Run Baseline CNN')
parser.add_argument('--balanced', action='store_true', help='Run the balanced dataset')
parser.add_argument('--semi', action='store_true', help='Run the semi-imbalanced dataset')
parser.add_argument('--inbalanced', action='store_true', help='Run the highly-imbalanced dataset')
parser.add_argument('--epochs', type=int, default = 40)

# the vq-vae specific dims
parser.add_argument('--code_book_dim', type=int, default=10, help="The number of discrete encoding vecotr")
parser.add_argument('--latent_channels', type = int, default = 10, help = "The dimension of the code book")
parser.add_argument('--hidden_dim', type=int, default = 10, help = "The intermediate hidden units")


args = parser.parse_args()
########################## LOADING DATASET ########################################################
print("Loading dataset...")
if args.balanced:
    test = 'balanced'
    print("Loading balanced dataset...")
    train_dataset = torch.load('expirement_data/balanced/balanced_dataset.pt', weights_only=False)
    # test_dataset = torch.load('expirement_data/balanced/test_dataset.pt', weights_only=False)
elif args.semi:
    test = 'semi'
    print("Loading Semi-Imbalanced dataset...")
    train_dataset = torch.load('expirement_data/semi/semi_imbalanced_dataset.pt', weights_only=False)
    # test_dataset = torch.load('expirement_data/semi/test_dataset.pt', weights_only=False    )
else:
    test = 'inbalanced'
    print("Loading Highly-Imbalanced dataset...")
    train_dataset = torch.load('expirement_data/high/highly_imbalanced_dataset.pt', weights_only=False)
    # test_dataset = torch.load('expirement_data/high/test_dataset.pt', weights_only=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
test_dataset = datasets.FashionMNIST(root=".data/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("Dataset loaded successfully")

########################################### EXPIREMENT SET UP ####################################################
print("Settings up expirement ...")
model = VQVAE(channel_in=784, hidden_dim=args.hidden_dim, 
              latent_channels=args.latent_channels, code_book_dim=args.code_book_dim)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)

############################################ TRAINING LOOP ##################################################
# for debugging and loss calc
recon_loss_log = []
qv_loss_log = []
test_recon_loss_log = []
train_loss = 0

for epoch in range(args.epochs):
    train_loss = 0
    model.train()
    for i, (x,y) in enumerate(tqdm(train_loader, leave=False, desc="Training")):
        x, y = x.to(device), y.to(device)
        
        recon_data, vq_loss, quantized = model(x.view(x.shape[0], -1))
        # put back into original shape
        recon_data = recon_data.view(recon_data.shape[0], 1, 28, 28)

        recon_loss = (recon_data - x).pow(2).mean()
        loss = vq_loss + recon_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recon_loss_log.append(recon_loss.item())
        qv_loss_log.append(vq_loss.item())
        train_loss += recon_loss.item()

print("Code has finished!")

# save the model to the results folder
torch.save(model.state_dict(), f'Results/VQVAE/vqvae_{test}.pth')

############################################ VISUALIZATION ##################################################
print("Generating visualizations...")

# Create Results/VQVAE directory if it doesn't exist
os.makedirs('Results/VQVAE', exist_ok=True)

# Set model to evaluation mode
model.eval()

# Get a batch of test images
test_batch, _ = next(iter(test_loader))
test_batch = test_batch.to(device)

# Generate reconstructions
with torch.no_grad():
    # Flatten the input for the VQ-VAE
    flat_input = test_batch.view(test_batch.shape[0], -1)
    recon_batch, _, _ = model(flat_input)
    # Reshape reconstruction back to image format
    recon_batch = recon_batch.view(recon_batch.shape[0], 1, 28, 28)

# Create a figure to display original and reconstructed images
plt.figure(figsize=(15, 5))

# Display original images
plt.subplot(1, 2, 1)
plt.title('Original Images')
plt.imshow(vutils.make_grid(test_batch[:8].cpu(), normalize=True).permute(1, 2, 0))
plt.axis('off')

# Display reconstructed images
plt.subplot(1, 2, 2)
plt.title('Reconstructed Images')
plt.imshow(vutils.make_grid(recon_batch[:8].cpu(), normalize=True).permute(1, 2, 0))
plt.axis('off')

# Save the visualization
plt.savefig(f'Results/VQVAE/vqvae_reconstruction_{test}.png')
plt.close()

# Plot training losses
plt.figure(figsize=(10, 5))
plt.plot(recon_loss_log, label='Reconstruction Loss')
plt.plot(qv_loss_log, label='VQ Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.savefig(f'Results/VQVAE/vqvae_losses_{test}.png')
plt.close()

print("Visualizations saved to Results/VQVAE/ directory")