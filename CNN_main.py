import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from ..models.VAE_CNN import CNN
from ..engine.engine import train_model, evaluate_cnn
import argparse

torch.serialization.add_safe_globals(["Subset"])

######################################### RUN THE CNN #########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Run Baseline CNN')
parser.add_argument('--balanced', action='store_true', help='Run the balanced dataset')
parser.add_argument('--semi', action='store_true', help='Run the semi-imbalanced dataset')
parser.add_argument('--inbalanced', action='store_true', help='Run the highly-imbalanced dataset')

args = parser.parse_args()
####################################### LOADING DATASET ######################################
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


################################ EXPERIMENT SETUP #################################
print("Setting up experiment...")
model = CNN()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

############################## CNN TRAINING #################################
print("Training CNN...")
train_model(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    num_epochs=40,
    input_dim=784,
    add_noise_flag=False,
    noise_factor=None,
    save_path=f"Results/Basline/cnn_model_full_dataset_{test}.pth",
    cnn_flag=True,
    noisy_classes=None
)
print("CNN training complete")


################################ Evaluate CNN #################################
print("Evaluating CNN...")
results_df = evaluate_cnn(
    model=model,
    test_loader=test_loader,
    device=device
)
# Save results to CSV
print(results_df)
results_df.to_csv(f'Results/Basline/cnn_results_full_dataset_{test}.csv', index=False)
print(f"Results saved to Results/Basline/cnn_results_full_dataset_{test}.csv")
print("Completed expirement!!")
