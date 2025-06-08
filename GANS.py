from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

class ConditionalGAN(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=512, channel_out=784, num_classes=10):  # Increased hidden_dim
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embedding layer for labels
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Generator - even deeper with residual connections
        self.generator = nn.Sequential(
            # Initial projection
            nn.Linear(latent_dim + num_classes, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            # Residual block 1
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            # Residual block 2
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            # Upscaling
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            
            # Final layer with pixel activation
            nn.Linear(hidden_dim, channel_out),
            nn.Sigmoid()
        )

        # Discriminator - spectral normalization for stability
        self.discriminator = nn.Sequential(
            # Feature extraction
            nn.utils.spectral_norm(nn.Linear(channel_out + num_classes, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Self-attention layer
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Feature compression
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Classification
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def generator_forward(self, z, labels):
        # Embed labels
        label_embedding = self.label_embedding(labels)
        # Concatenate noise and labels
        z = torch.cat([z, label_embedding], dim=1)
        return self.generator(z)

    def discriminator_forward(self, x, labels):
        # Embed labels
        label_embedding = self.label_embedding(labels)
        # Concatenate images and labels
        x = torch.cat([x, label_embedding], dim=1)
        return self.discriminator(x)

    def generate(self, num_samples, labels, device):
        """Generate images for specific labels"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.generator_forward(z, labels)

class BigGAN(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=512, num_classes=10, ch=64):
        super(BigGAN, self).__init__()
        self.latent_dim = latent_dim
        
        # Generator
        self.generator = nn.Sequential(
            # Initial dense layer
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            # Reshape to start convolutions
            # For 28x28 output, we start with 7x7
            nn.Linear(hidden_dim, 7 * 7 * ch * 8),
            nn.BatchNorm1d(7 * 7 * ch * 8),
            nn.ReLU(),
            
            # Will be reshaped to (batch_size, ch*8, 7, 7)
        )
        
        self.generator_conv = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(ch * 8, ch * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(ch * 4, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # 14x14 -> 7x7
            nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2),
            
            # 7x7 -> 1x1
            nn.Conv2d(ch * 2, 1, 7, stride=1, padding=0),
            nn.Sigmoid()
        )

    def generator_forward(self, z, labels):
        # One-hot encode labels
        labels_onehot = torch.zeros(labels.size(0), 10, device=labels.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        
        # Concatenate noise and labels
        z = torch.cat([z, labels_onehot], dim=1)
        
        # Forward through dense layers
        h = self.generator(z)
        
        # Reshape for convolutions: (batch_size, ch*8, 7, 7)
        h = h.view(h.size(0), -1, 7, 7)
        
        # Forward through conv layers
        return self.generator_conv(h)

    def discriminator_forward(self, x, labels):
        # One-hot encode labels
        labels_onehot = torch.zeros(labels.size(0), 10, device=labels.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        
        # Process the image through discriminator
        x = self.discriminator(x)
        
        # Combine with labels (you might want to adjust this part based on your needs)
        x = x.view(x.size(0), -1)  # Flatten
        return x.squeeze()

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=False, downsample=False):
        super().__init__()
        self.upsample = upsample
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
        
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        h = self.activation(self.bn1(x))
        
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        h = self.conv2(h)
        
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)
            
        return h + self.skip(x)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim//8, 1)
        self.key = nn.Conv2d(in_dim, in_dim//8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Compute attention scores
        q = self.query(x).view(B, -1, H*W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H*W)
        v = self.value(x).view(B, -1, H*W)
        
        attention = F.softmax(torch.bmm(q, k), dim=2)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        return self.gamma * out + x