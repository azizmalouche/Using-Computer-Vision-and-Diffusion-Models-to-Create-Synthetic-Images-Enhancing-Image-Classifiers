from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, z_dim=20):
        super().__init__()
        # encoding 
        self.img2hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden2hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2mu = nn.Linear(hidden_dim, z_dim)
        self.hidden2sigma = nn.Linear(hidden_dim, z_dim)

        # decoder 
        self.z2hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden2image = nn.Linear(hidden_dim, input_dim)
        
        # define ReLU
        self.relu = nn.ReLU()

    def encoder(self, x):
        h = self.relu(self.img2hidden(x))
        h = self.relu(self.hidden2hidden(h))
        mu = self.hidden2mu(h)
        logvar = self.hidden2sigma(h) 
        return mu, logvar
    
    def decoder(self, z):
        h = self.relu(self.z2hidden(z))
        h = self.relu(self.hidden2hidden(h))
        img = self.hidden2image(h)
        img = torch.sigmoid(img)
        return img


    def forward(self, x):
        mu, logvar = self.encoder(x)
        sigma = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma * epsilon

        x_reconstructed = self.decoder(z_reparameterized)

        return x_reconstructed, mu, logvar

class CNN(nn.Module):
    def __init__(self, image_dim=28, num_classes=10):
        super().__init__()
        self.input_len = int(64 * image_dim / 2 * image_dim / 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride = 1, padding=1, padding_mode='reflect')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,padding_mode='reflect')
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self.input_len, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.out(x)

        return x
    
# adding the VQ-VAE componenents 
class VectorQuantizer(nn.Module):
    # commitment cost is a hyperparameter 
    # part of the loss 
    def __init__(self, code_book_dim, embedding_dim, commitment_cost):
        super().__init__()
        self.code_book_dim = code_book_dim
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(code_book_dim, embedding_dim)
        # randomly initialize codebook
        self.embedding.weight.data.uniform_(-1/code_book_dim, 1/code_book_dim)

    def forward(self, inputs):
        # this flips the image from CxHxW to HxWxC
        # check if this hold for our problem
        # inputs = inputs.permute(0, 2, 3, 1).contiguous() # make sure this works for us PRINT THE SIZE WE WANT THIS TO BE BxHxWxC 
        # input_shape = inputs.shape

        # flat_input = inputs.view(-1, 1, self.embedding_dim)
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate the distance between each embedding and each codebook vector
        distances = torch.sum((flat_input.unsqueeze(1) - self.embedding.weight.unsqueeze(0)) ** 2, dim=2)


        # Find the closest codebook vector
        encoding_indices = torch.argmin(distances, dim=1)

        # get the quantized vector 
        quantized = self.embedding(encoding_indices)

        # Create loss that pulls encoder embeddings and codebook vector selected
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        q_latent_loss = F.mse_loss(quantized, flat_input.detach())
        #e_latent_loss = F.binary_cross_entropy(quantized.detach(), flat_input)
        #q_latent_loss = F.binary_cross_entropy(quantized, flat_input.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Reconstruct quantized representation using the encoder embeddings to allow for 
        # backpropagation of gradients into encoder
        if self.training:
            quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized, encoding_indices


class VQVAE(nn.Module):
    def __init__(self, channel_in, hidden_dim,latent_channels=32, code_book_dim=64, commitment_cost=0.25):
        super().__init__()
        # we will only define the Vector quantizer 
        self.vq = VectorQuantizer(code_book_dim=code_book_dim, 
                                  embedding_dim=latent_channels, 
                                  commitment_cost=commitment_cost)
        
        self.enc1 = nn.Linear(channel_in, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc3 = nn.Linear(hidden_dim, latent_channels)

        self.dec1 = nn.Linear(latent_channels, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec3 = nn.Linear(hidden_dim, channel_in)

        self.relu = nn.ReLU()

    def encode(self, x):
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        encoding = self.relu(self.enc3(x))

        vq_loss, quantized, encoding_indicies = self.vq(encoding)
        return vq_loss, quantized, encoding_indicies
    
    def decode(self, x):
        x = self.relu(self.dec1(x))
        x = self.relu(self.dec2(x))
        img = self.dec3(x)
        img = torch.sigmoid(img)
        return img
    
    def forward(self, x):
        vq_loss, quantized, encoding_indicies = self.encode(x)
        recon = self.decode(quantized)
        return recon, vq_loss, quantized






