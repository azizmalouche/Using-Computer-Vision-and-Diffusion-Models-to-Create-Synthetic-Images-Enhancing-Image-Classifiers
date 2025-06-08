import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm
from ..models.VAE_CNN import VariationalAutoEncoder, CNN
import numpy as np
import argparse
from engine import train_model, inference, evaluate_cnn

###################################### PARSING ARGUMENTS ######################################
parser = argparse.ArgumentParser(description='Train VAE model')
parser.add_argument('--input_dim', type=int, default=784, help='Input dimension')
parser.add_argument('--hidden_dim', type=int, default=200, help='Hidden dimension')
parser.add_argument('--z_dim', type=int, default=20, help='Latent dimension')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (Karpathy constant)')
parser.add_argument('--add_noise', action='store_true', help='Add noise to inputs')
parser.add_argument('--noise_factor', type=float, default=0.3, help='Noise factor')
parser.add_argument('--model_save_path', type=str, default='vae_model.pth', help='Path to save the model')
parser.add_argument('--inference_save_path', type=str, default='inference_reconstruction.png', help='Path to save the inference image')
parser.add_argument('--num_samples', type=int, default=8, help='Number of samples for inference')
parser.add_argument('--dataset_save_path', type=str, default='generated_dataset.pt', help='Path to save the generated dataset')
parser.add_argument('--CNN', action="store_true", help="Run the CNN for classification")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = args.input_dim
hidden_dim = args.hidden_dim
z_dim = args.z_dim
num_epochs = args.num_epochs
batch_size = args.batch_size
karpathy_constant = args.lr
add_noise_flag = args.add_noise
noise_factor = args.noise_factor
model_save_path = args.model_save_path
inference_save_path = args.inference_save_path
num_samples = args.num_samples  
dataset_save_path = args.dataset_save_path
cnn_flag = args.CNN

###################################### LOADING DATASET ######################################
print("LOADING THAT BITCHASS DATA IN")
dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
random_indices = np.random.choice(len(dataset), size=5000, replace=False)
subset_train_dataset = Subset(dataset, random_indices)
train_loader = DataLoader(dataset=subset_train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.FashionMNIST(root=".data/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

###################################### EXPERIMENT SETUP ######################################
if add_noise_flag:
    print(f"RUNNING DVAE BITCH")
elif cnn_flag:
    print("Running CNN")
else:
    print("RUNNING VAE")

if cnn_flag:
    model = CNN()
    loss_fn = nn.CrossEntropyLoss()
else:
    model = VariationalAutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, z_dim=z_dim)
    loss_fn = nn.BCELoss(reduction='sum')

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=karpathy_constant)

###################################### TRAINING ######################################
print("FINISHED TRAINING BABY")
noisy_classes = list(range(10)) #[1, 7, 8, 9]
train_model(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device,
    num_epochs=num_epochs,
    input_dim=input_dim,
    add_noise_flag=add_noise_flag,
    noise_factor=noise_factor,
    save_path=model_save_path,
    cnn_flag=cnn_flag,
    noisy_classes=noisy_classes if add_noise_flag else None  # Only pass noisy_classes if add_noise is True
)
print("DONE TRAINING!")

###################################### INFERENCE ######################################
print("GENERATING DATA")
inference(
    model=model,
    test_loader=test_loader,
    device=device,
    num_samples=num_samples,
    save_path=inference_save_path,
    dataset_save_path=dataset_save_path,
    cnn_flag=cnn_flag
)
print("FINISHED CODE")

# Evaluate CNN if the flag is set
if cnn_flag:
    accuracy = evaluate_cnn(model, test_loader, device)
    print(f"CNN Accuracy: {accuracy:.2f}")


# replicate their code (5000 samples)
# train the CNN (train/test split) and see how it does. Use the same train/test split for all the models.
# train VAE on training data only and produce 500 more images, feed that into CNN and see how it does. Then have it produce 1000,15000,etc, etc and see how CNN improves. add a function.
# Then do the same with DVAE.
# try adding different types of noise to dvae (not just gaussian). can include augmentation as well. 

# do same with vqae once azfals is done
# see if we can add clip to cart part
#gg
