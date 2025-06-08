import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os
from diffusers import UNet2DConditionModel, DDPMScheduler, UNet2DModel
from torchvision.transforms import functional as TF
from VAE import VariationalAutoEncoder, CNN, VQVAE
from GANS import ConditionalGAN, BigGAN
from engine import train_model, train_gan, train_vqvae, train_biggan
from inference_plot import evaluate_cnn

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Noise configurations
noise_configs = [
    ['gaussian'],
    ['salt_pepper'],
    ['rotation'],
    ['brightness'],
    ['contrast'],
    ['blur'],
]

def evaluate_cnn(cnn_model, test_loader):
    """Evaluate CNN performance"""
    cnn_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = cnn_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_cnn_on_generated_images(generated_images, generated_labels, device, num_epochs=10, batch_size=128, lr=0.001):
    """Train CNN on generated images and return the trained model."""
    # Ensure input tensors are on the correct device
    generated_images = generated_images.to(device)
    generated_labels = generated_labels.to(device)

    cnn_model = CNN().to(device)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining CNN on generated images...")
    for epoch in range(num_epochs):
        cnn_model.train()
        running_loss = 0.0
        for i in range(0, len(generated_images), batch_size):
            batch_images = generated_images[i:i+batch_size].to(device)
            batch_labels = generated_labels[i:i+batch_size].to(device)

            optimizer.zero_grad()
            outputs = cnn_model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"CNN Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(generated_images):.4f}")

    return cnn_model

def train_and_evaluate(dataset_type, dataset, size):
    """Train and evaluate CNN on VAE and DVAE with different noise types"""
    results = []

    # Create organized directory structure
    base_path = '/content/drive/MyDrive/Colab Notebooks/CV_Final Project/VAE_vs_Others/AZIZ'
    if dataset_type == 'balanced':
        base_dir = f'{base_path}/Balanced'
    elif dataset_type == 'high':
        base_dir = f'{base_path}/Highly Imbalanced'
    elif dataset_type == 'semi':
        base_dir = f'{base_path}/Semi Imbalanced'

    # Create subdirectories for models and images
    model_dir = f'{base_dir}/models/{size}'
    image_dir = f'{base_dir}/images/{size}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    print(f"\nSaving models to: {model_dir}")
    print(f"Saving images to: {image_dir}")

    # Update save paths throughout the code
    vae_path = f'{model_dir}/vae.pth'
    gan_path = f'{model_dir}/gan.pth'
    biggan_path = f'{model_dir}/biggan.pth'
    vqvae_path = f'{model_dir}/vqvae.pth'

    # Update image save paths
    vae_recon_path = f'{image_dir}/vae_reconstructions.png'
    gan_recon_path = f'{image_dir}/gan_reconstructions.png'
    biggan_recon_path = f'{image_dir}/biggan_reconstructions.png'
    vqvae_recon_path = f'{image_dir}/vqvae_reconstructions.png'

    # Keep aggregate results in base directory
    plot_path = f'{base_dir}/cnn_comparison_{size}.png'

    print(f"\n{'='*50}")
    print(f"Starting experiments with {dataset_type.upper()} dataset")
    print(f"{'='*30}")

    test_dataset = datasets.FashionMNIST(root=".data/", train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    print(f"Test dataset loaded. Size: {len(test_dataset)} images")

    # Create data loaders
    train_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)

    # Get fixed set of images - one from each class
    fixed_images = []
    fixed_labels = []
    for class_idx in range(10):
        for images, labels in train_loader:
            images = images.to(device)  # Ensure images are on the correct device
            labels = labels.to(device)  # Ensure labels are on the correct device
            class_mask = (labels == class_idx)
            if any(class_mask):
                fixed_images.append(images[class_mask][0:1])
                fixed_labels.append(labels[class_mask][0:1])
                break

    fixed_images = torch.cat(fixed_images).to(device)
    fixed_labels = torch.cat(fixed_labels).to(device)

    #===========ORIGINAL IMAGES===========
    print(f"\n{'='*30}")
    print("Training CNN on original images...")
    print(f"{'='*30}")

    # Convert train_loader data into tensors for CNN training
    original_images = []
    original_labels = []
    for images, labels in train_loader:
        original_images.append(images.to(device))
        original_labels.append(labels.to(device))
    
    original_images = torch.cat(original_images)
    original_labels = torch.cat(original_labels)

    # Train CNN on full training dataset
    cnn_model = train_cnn_on_generated_images(
        generated_images=original_images,
        generated_labels=original_labels,
        device=device
    )
    
    print("\nEvaluating CNN on test set...")
    accuracy = evaluate_cnn(cnn_model, test_loader)
    results.append(('Original', accuracy))
    print(f"Original Images Accuracy: {accuracy:.2f}%")


    #===========Original Images with Noise/Augmentation===========
    noise_og_images = ['gaussian', 'rotation', 'blur']
    print(f"\n{'='*30}")
    print("Training CNN on noisy/augmented original images...")
    print(f"{'='*30}")

    for noise_type in noise_og_images:
        print(f"\nApplying {noise_type} noise/augmentation...")
        
        noisy_images = []
        noisy_labels = []
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            if noise_type == 'gaussian':
                # Add Gaussian noise
                noise = torch.randn_like(images) * 0.3
                noisy = torch.clamp(images + noise, 0., 1.)

            elif noise_type == 'rotation':
                # Apply rotation augmentation
                angle = 0.3 * 30  # Using noise_factor of 0.1 like gaussian noise above
                noisy = TF.rotate(images, angle)
            elif noise_type == 'blur':
                # Apply Gaussian blur
                kernel_size = int(3 + 0.3 * 4)
                kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
                noisy = TF.gaussian_blur(images, kernel_size=kernel_size, sigma=0.3)
            
            noisy_images.append(noisy)
            noisy_labels.append(labels)
        
        noisy_images = torch.cat(noisy_images)
        noisy_labels = torch.cat(noisy_labels)
        
        # Train CNN on noisy/augmented images
        print(f"\nTraining CNN on {noise_type} noisy/augmented images...")
        cnn_model = train_cnn_on_generated_images(
            generated_images=noisy_images,
            generated_labels=noisy_labels,
            device=device
        )
        
        print("\nEvaluating CNN on test set...")
        accuracy = evaluate_cnn(cnn_model, test_loader)
        results.append((f'Original with {noise_type}', accuracy))
        print(f"Original with {noise_type} Accuracy: {accuracy:.2f}%")


    #========DIFFUSION========
    print(f"\n{'='*30}")
    print(f"Loading Diffusion model for {dataset_type} dataset...")
    print(f"{'='*30}")

    # Load in diffusion model
    diffusion_model = UNet2DModel(
        sample_size=28,           # Image size
        in_channels=1,            # Grayscale input
        out_channels=1,           # Grayscale output
        layers_per_block=2,       # Number of layers per block
        block_out_channels=(64, 128, 256),  # Channel dimensions
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        num_class_embeds=10,  # Add class conditioning
    ).to(device)

    diffusion_path = f'{model_dir}/diffusion_{dataset_type}_{size}.pth'

    if os.path.exists(diffusion_path):
        print(f"Loading Diffusion model from {diffusion_path}")
        # Load model state dict directly to the device
        diffusion_model.load_state_dict(
            torch.load(diffusion_path, map_location=device)
        )
        diffusion_model.eval()

        # Get reconstructions of fixed images
        scheduler = DDPMScheduler(num_train_timesteps=250, beta_schedule="squaredcos_cap_v2")

        @torch.no_grad()
        def sample_from_diffusion(model, scheduler, images, labels, device):
            # Start with random noise
            noisy_images = torch.randn_like(images).to(device)
            
            print("\nStarting diffusion sampling...")
            total_steps = scheduler.config.num_train_timesteps
            # Gradually denoise
            for t in reversed(range(total_steps)):
                if t % 20 == 0:  # Print progress every 20 steps
                    print(f"Sampling step {total_steps-t}/{total_steps}")
                    
                timesteps = torch.tensor([t], device=device)
                timesteps = timesteps.expand(images.shape[0])
                
                # Add class labels to forward pass
                model_output = model(noisy_images, timesteps, class_labels=labels).sample
                noisy_images = scheduler.step(model_output, t, noisy_images).prev_sample
            
            return noisy_images

        print("\nGenerating fixed image reconstructions...")
        fixed_reconstructions = sample_from_diffusion(diffusion_model, scheduler, fixed_images, fixed_labels, device)

        # Create and save comparison visualization
        comparison = torch.cat([fixed_images, fixed_reconstructions])
        vutils.save_image(comparison,
                         f'{image_dir}/diffusion_reconstructions_{dataset_type}.png',
                         nrow=10,
                         normalize=False,
                         padding=2)
        print(f"Saved reconstruction comparison to {image_dir}/diffusion_reconstructions_{dataset_type}.png")

        # Generate full dataset reconstructions AND include original images for CNN training
        generated_images = []
        generated_labels = []

        print("\nGenerating samples from diffusion model...")
        total_batches = len(train_loader)
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(train_loader):
                print(f"Processing batch {batch_idx+1}/{total_batches}")
                data = data.to(device)
                labels = labels.to(device)
                samples = sample_from_diffusion(diffusion_model, scheduler, data, labels, device)

                # Add both original and generated images
                generated_images.append(data)
                generated_images.append(samples)
                generated_labels.append(labels)
                generated_labels.append(labels)

        generated_images = torch.cat(generated_images).to(device)
        generated_labels = torch.cat(generated_labels).to(device)
        print(f"Total images: {len(generated_images)}")

        # Train CNN
        cnn_model = train_cnn_on_generated_images(generated_images, generated_labels, device)
        print("\nEvaluating CNN on test set...")
        accuracy = evaluate_cnn(cnn_model, test_loader)
        results.append(('Diffusion', accuracy))
        print(f"Diffusion Accuracy: {accuracy:.2f}%")
    else:
        print(f"No diffusion model found at {diffusion_path}")

    # For each Diffusion noise type:
    save_generated_images(
        original_images=data,
        generated_images=generated_images,
        original_labels=labels,
        generated_labels=generated_labels,
        model_name=f'Diffusion',
        dataset_type=dataset_type,
        size=size
    )

    #========CLIP GUIDED DIFFUSION========
    print(f"\n{'='*30}")
    print(f"Loading CLIP-guided Diffusion model for {dataset_type} dataset...")
    print(f"{'='*30}")

    # Load in the diffusion model
    clip_diffusion_model = UNet2DConditionModel(
        sample_size=28,           # Image size
        in_channels=1,            # Grayscale input
        out_channels=1,           # Grayscale output
        layers_per_block=2,       # Keep original layers
        block_out_channels=(64, 128, 256),  # Match your saved model
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        cross_attention_dim=512  # CLIP embedding dimension
    ).to(device)  # Ensure model is on the correct device

    clip_diffusion_path = f'{model_dir}/clip_diffusion_{dataset_type}_{size}.pth'

    if os.path.exists(clip_diffusion_path):
        print(f"Loading CLIP-guided Diffusion model from {clip_diffusion_path}")
        # Load model state dict directly to the device
        clip_diffusion_model.load_state_dict(
            torch.load(clip_diffusion_path, map_location=device)
        )
        clip_diffusion_model.eval()

        # Get reconstructions of fixed images
        scheduler = DDPMScheduler(num_train_timesteps=250, beta_schedule="squaredcos_cap_v2")

        @torch.no_grad()
        def sample_from_clip_diffusion(model, scheduler, images, device):
            # Ensure images are on the correct device
            images = images.to(device)
            null_embedding = torch.zeros(images.shape[0], 512, device=device)
            noisy_images = torch.randn_like(images).to(device)

            # Gradually denoise
            for t in reversed(range(scheduler.config.num_train_timesteps)):
                # Make sure timesteps is a single integer tensor
                timesteps = torch.tensor([t], device=device)

                # Expand timesteps for batch
                timesteps = timesteps.expand(images.shape[0])

                model_output = model(
                    noisy_images,
                    timesteps,
                    encoder_hidden_states=null_embedding.unsqueeze(1)
                ).sample

                noisy_images = scheduler.step(
                    model_output,
                    t,  # Use integer t instead of tensor timesteps
                    noisy_images
                ).prev_sample

            return noisy_images

        fixed_reconstructions = sample_from_clip_diffusion(clip_diffusion_model, scheduler, fixed_images, device)

        # Create and save comparison visualization
        comparison = torch.cat([fixed_images, fixed_reconstructions])
        vutils.save_image(comparison,
                         f'{image_dir}/clip_diffusion_reconstructions_{dataset_type}.png',
                         nrow=10,
                         normalize=False,
                         padding=2)
        print(f"Saved reconstruction comparison to {image_dir}/clip_diffusion_reconstructions_{dataset_type}.png")

        # Generate full dataset reconstructions AND include original images for CNN training
        generated_images = []
        generated_labels = []

        print("Generating samples from CLIP-guided diffusion model...")
        with torch.no_grad():
            for data, labels in train_loader:
                data = data.to(device)  # Ensure data is on the correct device
                labels = labels.to(device)  # Ensure labels are on the correct device
                samples = sample_from_clip_diffusion(clip_diffusion_model, scheduler, data, device)

                # Add both original and generated images
                generated_images.append(data)
                generated_images.append(samples)
                generated_labels.append(labels)
                generated_labels.append(labels)

        generated_images = torch.cat(generated_images).to(device)
        generated_labels = torch.cat(generated_labels).to(device)
        print(f"Total images: {len(generated_images)}")

        # Train CNN
        cnn_model = train_cnn_on_generated_images(generated_images, generated_labels, device)
        print("\nEvaluating CNN on test set...")
        accuracy = evaluate_cnn(cnn_model, test_loader)
        results.append(('CLIP-guided Diffusion', accuracy))
        print(f"CLIP-guided Diffusion Accuracy: {accuracy:.2f}%")
    else:
        print(f"No CLIP-guided diffusion model found at {clip_diffusion_path}")

    # For each CLIP-guided Diffusion noise type:
    save_generated_images(
        original_images=data,
        generated_images=generated_images,
        original_labels=labels,
        generated_labels=generated_labels,
        model_name=f'CLIP-guided Diffusion',
        dataset_type=dataset_type,
        size=size
    )

    #===========BigGAN===========
    print(f"\n{'='*30}")
    print(f"Training BigGAN on {dataset_type} dataset...")
    print(f"{'='*30}")

    biggan_model = BigGAN(
        latent_dim=128,
        hidden_dim=512,
        num_classes=10,
        ch=64
    ).to(device)  # Ensure model is on the correct device

    biggan_path = f'{model_dir}/biggan.pth'

    if os.path.exists(biggan_path):
        print(f"Loading BigGAN model from {biggan_path}")
        # Load model state dict directly to the device
        biggan_model.load_state_dict(torch.load(biggan_path, map_location=device))
    else:
        print("Training new BigGAN model...")
        optimizer_G = torch.optim.Adam(
            biggan_model.generator.parameters(),
            lr=1e-4,
            betas=(0.0, 0.999)
        )
        optimizer_D = torch.optim.Adam(
            biggan_model.discriminator.parameters(),
            lr=4e-4,
            betas=(0.0, 0.999)
        )

        train_biggan(
            model=biggan_model,
            train_loader=train_loader,
            optimizers=(optimizer_G, optimizer_D),
            device=device,
            num_epochs=50
        )

        # Save BigGAN model
        print(f"Saving BigGAN model as {biggan_path}")
        torch.save(biggan_model.state_dict(), biggan_path)

    print("\nBigGAN training completed. Generating samples...")

    # First generate samples for visualization with fixed labels
    with torch.no_grad():
        z = torch.randn(len(fixed_images), biggan_model.latent_dim).to(device)  # Ensure z is on the correct device
        generated_samples = biggan_model.generator_forward(z, fixed_labels)

        # Create and save comparison visualization
        comparison = torch.cat([fixed_images, generated_samples])
        vutils.save_image(comparison,
                         f'{image_dir}/biggan_reconstructions.png',
                         nrow=10,
                         normalize=False,
                         padding=2)
        print(f"Saved reconstruction comparison to {image_dir}/biggan_reconstructions.png")

    # Then generate samples for CNN training
    generated_images = []
    generated_labels = []

    with torch.no_grad():
        for data, labels in train_loader:
            batch_size = data.size(0)
            data = data.to(device)  # Ensure data is on the correct device
            labels = labels.to(device)  # Ensure labels are on the correct device

            # Generate fake images
            z = torch.randn(batch_size, biggan_model.latent_dim).to(device)  # Ensure z is on the correct device
            fake_images = biggan_model.generator_forward(z, labels)

            # Add both original and generated
            generated_images.append(data)
            generated_images.append(fake_images)
            generated_labels.append(labels)
            generated_labels.append(labels)

    generated_images = torch.cat(generated_images).to(device)
    generated_labels = torch.cat(generated_labels).to(device)
    print(f"Total images: {len(generated_images)}")

    # Train CNN
    cnn_model = train_cnn_on_generated_images(generated_images, generated_labels, device)
    print("\nEvaluating CNN on test set...")
    accuracy = evaluate_cnn(cnn_model, test_loader)
    results.append(('BigGAN', accuracy))
    print(f"BigGAN Accuracy: {accuracy:.2f}%")

    # For each BIGGAN noise type:
    save_generated_images(
        original_images=data,
        generated_images=generated_images,
        original_labels=labels,
        generated_labels=generated_labels,
        model_name=f'BIGGAN',
        dataset_type=dataset_type,
        size=size
    )

    #===========GAN===========
    print(f"\n{'='*30}")
    print(f"Training/Loading GAN for {dataset_type} dataset...")
    print(f"{'='*30}")

    gan_model = ConditionalGAN(latent_dim=100, hidden_dim=200, channel_out=784, num_classes=10).to(device)  # Ensure model is on the correct device
    gan_path = f'{model_dir}/gan.pth'

    if os.path.exists(gan_path):
        print(f"Loading GAN model from {gan_path}")
        # Load model state dict directly to the device
        gan_model.load_state_dict(torch.load(gan_path, map_location=device))
    else:
        print("Training new GAN model...")
        optimizer_G = torch.optim.Adam(gan_model.generator.parameters(), lr=3e-4)
        optimizer_D = torch.optim.Adam(gan_model.discriminator.parameters(), lr=3e-4)
        criterion = nn.BCELoss(reduction='sum')

        train_gan(
            model=gan_model,
            train_loader=train_loader,
            optimizers=(optimizer_G, optimizer_D),
            criterion=criterion,
            device=device,
            num_epochs=50
        )

        # Save GAN model
        print(f"Saving GAN model as {gan_path}")
        torch.save(gan_model.state_dict(), gan_path)

    print("\nGAN training completed. Generating reconstructions...")

    # Generate samples for visualization
    with torch.no_grad():
        # Pass fixed_labels along with the number of samples
        generated_samples = gan_model.generate(len(fixed_images), fixed_labels, device)
        generated_samples = generated_samples.view(-1, 1, 28, 28)

    # Create and save comparison visualization
    comparison = torch.cat([fixed_images, generated_samples])
    vutils.save_image(comparison,
                     f'{image_dir}/gan_reconstructions.png',
                     nrow=10,
                     normalize=False,
                     padding=2)
    print(f"Saved reconstruction comparison to {image_dir}/gan_reconstructions.png")

    # Generate full dataset reconstructions for CNN training
    generated_images = []
    generated_labels = []

    with torch.no_grad():
        for data, labels in train_loader:
            batch_size = data.size(0)
            data = data.to(device)  # Ensure data is on the correct device
            labels = labels.to(device)  # Ensure labels are on the correct device
            noise = torch.randn(batch_size, gan_model.latent_dim).to(device)  # Ensure noise is on the correct device
            fake_images = gan_model.generator_forward(noise, labels)
            fake_images = fake_images.view(-1, 1, 28, 28)

            # Add both original and reconstructed images
            generated_images.append(data)
            generated_images.append(fake_images)
            generated_labels.append(labels)
            generated_labels.append(labels)

    generated_images = torch.cat(generated_images).to(device)
    generated_labels = torch.cat(generated_labels).to(device)
    print(f"Total images: {len(generated_images)}")

    # Train CNN
    cnn_model = train_cnn_on_generated_images(generated_images, generated_labels, device)
    print("\nEvaluating CNN on test set...")
    accuracy = evaluate_cnn(cnn_model, test_loader)
    results.append(('GAN', accuracy))
    print(f"GAN Accuracy: {accuracy:.2f}%")

    # For each GAN noise type:
    save_generated_images(
        original_images=data,
        generated_images=generated_images,
        original_labels=labels,
        generated_labels=generated_labels,
        model_name=f'GAN',
        dataset_type=dataset_type,
        size=size
    )

    #===========VQVAE===========
    print(f"\n{'='*30}")
    print(f"Training/Loading VQ-VAE for {dataset_type} dataset...")
    print(f"{'='*30}")

    vqvae_model = VQVAE(channel_in=784, hidden_dim=200, latent_channels=20, code_book_dim=256).to(device)  # Ensure model is on the correct device
    vqvae_path = f'{model_dir}/vqvae.pth'

    if os.path.exists(vqvae_path):
        print(f"Loading VQVAE model from {vqvae_path}")
        # Load model state dict directly to the device
        vqvae_model.load_state_dict(torch.load(vqvae_path, map_location=device))
    else:
        print("Training new VQVAE model...")
        optimizer = torch.optim.Adam(vqvae_model.parameters(), lr=3e-4)

        train_vqvae(
            model=vqvae_model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=50
        )

        # Save VQVAE model
        print(f"Saving VQVAE model as {vqvae_path}")
        torch.save(vqvae_model.state_dict(), vqvae_path)

    print("\nVQ-VAE training completed. Generating reconstructions...")

    # Get reconstructions of fixed images
    with torch.no_grad():
        fixed_reconstructions, _, _ = vqvae_model(fixed_images.view(-1, 784))
        fixed_reconstructions = fixed_reconstructions.view(-1, 1, 28, 28)

    # Create and save comparison visualization using fixed images and their reconstructions
    comparison = torch.cat([fixed_images, fixed_reconstructions])
    vutils.save_image(comparison,
                     f'{image_dir}/vqvae_reconstructions.png',
                     nrow=10,
                     normalize=False,
                     padding=2)
    print(f"Saved reconstruction comparison to {image_dir}/vqvae_reconstructions.png")

    # Generate full dataset reconstructions for CNN training
    generated_images = []
    generated_labels = []

    with torch.no_grad():
        for data, labels in train_loader:
            data = data.to(device)  # Ensure data is on the correct device
            labels = labels.to(device)  # Ensure labels are on the correct device

            recon, _, _ = vqvae_model(data.view(-1, 784))
            recon = recon.view(-1, 1, 28, 28)

            generated_images.append(data)
            generated_images.append(recon)
            generated_labels.append(labels)
            generated_labels.append(labels)

    generated_images = torch.cat(generated_images).to(device)
    generated_labels = torch.cat(generated_labels).to(device)
    print(f"Total images: {len(generated_images)}")

    # Train CNN
    cnn_model = train_cnn_on_generated_images(generated_images, generated_labels, device)
    print("\nEvaluating CNN on test set...")
    accuracy = evaluate_cnn(cnn_model, test_loader)
    results.append(('VQ-VAE', accuracy))
    print(f"VQ-VAE Accuracy: {accuracy:.2f}%")

    # For each VQ-VAE noise type:
    save_generated_images(
        original_images=data,
        generated_images=generated_images,
        original_labels=labels,
        generated_labels=generated_labels,
        model_name=f'VQVAE',
        dataset_type=dataset_type,
        size=size
    )

    #===========VAE===========
    print(f"\n{'='*30}")
    print(f"Training/Loading VAE for {dataset_type} dataset...")
    print(f"{'='*30}")

    vae_model = VariationalAutoEncoder(input_dim=784, hidden_dim=200, z_dim=20).to(device)  # Ensure model is on the correct device
    vae_path = f'{model_dir}/vae.pth'

    if os.path.exists(vae_path):
        print(f"Loading VAE model from {vae_path}")
        # Load model state dict directly to the device
        vae_model.load_state_dict(torch.load(vae_path, map_location=device))
    else:
        print("Training new VAE model...")
        optimizer = torch.optim.Adam(vae_model.parameters(), lr=3e-4)

        train_model(
            model=vae_model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=nn.BCELoss(reduction='sum'),
            device=device,
            num_epochs=50,
            input_dim=784,
            add_noise_flag=False
        )

        # Save VAE model
        print(f"Saving VAE model as {vae_path}")
        torch.save(vae_model.state_dict(), vae_path)

    print("\nVAE training completed. Generating reconstructions...")

    # Get reconstructions of fixed images
    with torch.no_grad():
        fixed_reconstructions, _, _ = vae_model(fixed_images.view(-1, 784))
        fixed_reconstructions = fixed_reconstructions.view(-1, 1, 28, 28)

    # Create and save comparison visualization using fixed images and their reconstructions
    comparison = torch.cat([fixed_images, fixed_reconstructions])
    vutils.save_image(comparison,
                     f'{image_dir}/vae_reconstructions.png',
                     nrow=10,
                     normalize=False,
                     padding=2)
    print(f"Saved reconstruction comparison to {image_dir}/vae_reconstructions.png")

    # Generate full dataset reconstructions AND include original images for CNN training
    generated_images = []
    generated_labels = []

    with torch.no_grad():
        for data, labels in train_loader:
            data = data.to(device)  # Ensure data is on the correct device
            labels = labels.to(device)  # Ensure labels are on the correct device

            # Get reconstructions
            recon, _, _ = vae_model(data.view(-1, 784))
            recon = recon.view(-1, 1, 28, 28)

            # Add both original and reconstructed images
            generated_images.append(data)        # Original images
            generated_images.append(recon)       # Reconstructed images
            generated_labels.append(labels)      # Original labels
            generated_labels.append(labels)      # Same labels for reconstructions

    generated_images = torch.cat(generated_images).to(device)
    generated_labels = torch.cat(generated_labels).to(device)
    print(f"Total images: {len(generated_images)}")

    # Train CNN
    cnn_model = train_cnn_on_generated_images(generated_images, generated_labels, device)
    print("\nEvaluating CNN on test set...")
    accuracy = evaluate_cnn(cnn_model, test_loader)
    results.append(('VAE', accuracy))
    print(f"VAE Accuracy: {accuracy:.2f}%")

    # For each VAE noise type:
    save_generated_images(
        original_images=data,
        generated_images=generated_images,
        original_labels=labels,
        generated_labels=generated_labels,
        model_name=f'VAE',
        dataset_type=dataset_type,
        size=size
    )

    #===========DVAE===========
    for noise_type in noise_configs:
        noise_name = noise_type[0]
        print(f"\n{'='*30}")
        print(f"Training/Loading DVAE with {noise_name} on {dataset_type} dataset...")
        print(f"{'='*30}")

        dvae_model = VariationalAutoEncoder(input_dim=784, hidden_dim=200, z_dim=20).to(device)  # Ensure model is on the correct device
        dvae_path = f'{model_dir}/dvae_{noise_name}.pth'

        if os.path.exists(dvae_path):
            print(f"Loading DVAE model from {dvae_path}")
            # Load model state dict directly to the device
            dvae_model.load_state_dict(torch.load(dvae_path, map_location=device))
        else:
            print(f"Training new DVAE model with {noise_name} noise...")
            optimizer = torch.optim.Adam(dvae_model.parameters(), lr=3e-4)

            train_model(
                model=dvae_model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_fn=nn.BCELoss(reduction='sum'),
                device=device,
                num_epochs=50,
                input_dim=784,
                add_noise_flag=True,
                noise_types=noise_type,
                noisy_classes=list(range(10))
            )

            # Save DVAE model
            print(f"Saving DVAE model as {dvae_path}")
            torch.save(dvae_model.state_dict(), dvae_path)

        # Get reconstructions of fixed images
        with torch.no_grad():
            fixed_reconstructions, _, _ = dvae_model(fixed_images.view(-1, 784))
            fixed_reconstructions = fixed_reconstructions.view(-1, 1, 28, 28)

        # Create and save comparison visualization
        comparison = torch.cat([fixed_images, fixed_reconstructions])
        vutils.save_image(comparison,
                         f'{image_dir}/dvae_reconstructions_{noise_name}.png',
                         nrow=10,
                         normalize=False,
                         padding=2)
        print(f"Saved reconstruction comparison to {image_dir}/dvae_reconstructions_{noise_name}.png")

        # Generate full dataset reconstructions AND include original images for CNN training
        generated_images = []
        generated_labels = []

        with torch.no_grad():
            for data, labels in train_loader:
                data = data.to(device)  # Ensure data is on the correct device
                labels = labels.to(device)  # Ensure labels are on the correct device

                recon, _, _ = dvae_model(data.view(-1, 784))
                recon = recon.view(-1, 1, 28, 28)

                # Add both original and reconstructed images
                generated_images.append(data)        # Original images
                generated_images.append(recon)       # Reconstructed images
                generated_labels.append(labels)      # Original labels
                generated_labels.append(labels)      # Same labels for reconstructions

        generated_images = torch.cat(generated_images).to(device)
        generated_labels = torch.cat(generated_labels).to(device)
        print(f"Total images: {len(generated_images)}")

        # Train CNN
        cnn_model = train_cnn_on_generated_images(generated_images, generated_labels, device)
        print("\nEvaluating CNN on test set...")
        accuracy = evaluate_cnn(cnn_model, test_loader)
        results.append((f'DVAE ({noise_name})', accuracy))
        print(f"DVAE ({noise_name}) Accuracy: {accuracy:.2f}%")

        # For each DVAE noise type:
        save_generated_images(
            original_images=data,
            generated_images=generated_images,
            original_labels=labels,
            generated_labels=generated_labels,
            model_name=f'DVAE ({noise_name})',
            dataset_type=dataset_type,
            size=size
        )

    #===========PLOT===========
    print(f"\nCreating comparison plot for {dataset_type} dataset...")
    # Plot results
    plt.figure(figsize=(12, 6))
    names, accs = zip(*results)

    # Define color map (same as the one used for line plots)
    color_map = {
        'Original': '#1f77b4',      # blue
        'VAE': '#2ca02c',          # green
        'VQ-VAE': '#ff7f0e',       # orange
        'BigGAN': '#d62728',       # red
        'GAN': '#9467bd',          # purple
        # Original augmentations
        'Original with gaussian': '#17becf',  # cyan
        'Original with rotation': '#bcbd22',  # yellow-green
        'Original with blur': '#c7c7c7',      # light gray
        # All DVAE variants
        'DVAE (gaussian)': '#8c564b',     # brown
        'DVAE (rotation)': '#e377c2',     # pink
        'DVAE (blur)': '#7f7f7f',         # gray
        'DVAE (salt_pepper)': '#2f4f4f',  # dark slate gray
        'DVAE (brightness)': '#deb887',   # burlywood
        'DVAE (contrast)': '#6495ed',     # cornflower blue
        # Diffusion
        'Diffusion': '#8B008B',           # dark magenta
        'CLIP-guided Diffusion': '#20B2AA', # light sea green
    }

    # Get colors for each bar
    colors = [color_map.get(name, '#17becf') for name in names]
    bars = plt.bar(names, accs, color='#1f77b4')

    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Model Type')
    plt.ylabel('CNN Accuracy (%)')
    plt.title(f'CNN Performance: \nTrained on {dataset_type} dataset (size={size})')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')

    plt.tight_layout()
    # Save plot in appropriate directory
    print(f"Saving plot as {plot_path}")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

    return results

def save_generated_images(original_images, generated_images, original_labels, generated_labels, model_name, dataset_type, size):
    """Save both original and generated images as .pt file with detailed filename including all_images"""
    # Create base directory path
    base_path = '/content/drive/MyDrive/Colab Notebooks/CV_Final Project/VAE_vs_Others/AZIZ'
    
    # Map dataset_type to folder name
    dataset_folders = {
        'balanced': 'Balanced',
        'high': 'Highly Imbalanced',
        'semi': 'Semi Imbalanced'
    }
    
    # Create full path: AZIZ/[Balance Type]/all_images/[size]
    dataset_dir = os.path.join(base_path, dataset_folders[dataset_type])
    images_dir = os.path.join(dataset_dir, 'all_images', str(size))
    
    # Create all necessary directories
    os.makedirs(images_dir, exist_ok=True)  # This will create all parent directories if they don't exist
    
    # Combine original and generated data
    all_images = torch.cat([original_images, generated_images])
    all_labels = torch.cat([original_labels, generated_labels])
    
    # Create detailed filename including all_images
    filename = f'{model_name}_all_images_{dataset_type}_{size}.pt'
    save_path = os.path.join(images_dir, filename)
    
    torch.save({
        'images': all_images,
        'labels': all_labels,
        'model': model_name,
        'dataset_type': dataset_type,
        'size': size
    }, save_path)
    print(f"Saved combined dataset to {save_path}")

if __name__ == "__main__":
    # Create Results directory
    os.makedirs('Results', exist_ok=True)

    # Track results for plotting
    all_results = {
        'balanced': {},
        'semi': {},
        'high': {}
    }

    # Run for all dataset types and sizes
    dataset_sizes = [1000, 2000, 3000, 4000, 5000]

    for dataset_type in ['balanced', 'semi', 'high']:
        all_results[dataset_type] = {size: [] for size in dataset_sizes}

        for size in dataset_sizes:
            print(f"\n{'='*50}")
            print(f"Processing {dataset_type.upper()} dataset with {size} samples")
            print(f"{'='*50}")

            # Load appropriate dataset
            if dataset_type == 'balanced':
                dataset_path = f'/content/drive/MyDrive/Colab Notebooks/CV_Final Project/VAE_vs_Others/balanced_{size}.pt'
            elif dataset_type == 'semi':
                dataset_path = f'/content/drive/MyDrive/Colab Notebooks/CV_Final Project/VAE_vs_Others/semi_imbalanced_{size}.pt'
            else:  # high
                dataset_path = f'/content/drive/MyDrive/Colab Notebooks/CV_Final Project/VAE_vs_Others/highly_imbalanced_{size}.pt'

            dataset = torch.load(dataset_path, weights_only=False)
            results = train_and_evaluate(dataset_type, dataset, size)
            all_results[dataset_type][size] = results

    # Create comparison plots for each dataset type
    for dataset_type in ['balanced', 'semi', 'high']:
        plt.figure(figsize=(12, 8))

        # Extract results for each model type
        model_results = {}
        for size in dataset_sizes:
            for model_name, acc in all_results[dataset_type][size]:
                if model_name not in model_results:
                    model_results[model_name] = []
                model_results[model_name].append(acc)

        # Define a color map for each model type
        color_map = {
            'Original': '#1f77b4',      # blue
            'VAE': '#2ca02c',          # green
            'VQ-VAE': '#ff7f0e',       # orange
            'BigGAN': '#d62728',       # red
            'GAN': '#9467bd',          # purple
            # Original augmentations
            'Original with gaussian': '#17becf',  # cyan
            'Original with rotation': '#bcbd22',  # yellow-green
            'Original with blur': '#c7c7c7',      # light gray
            # All DVAE variants
            'DVAE (gaussian)': '#8c564b',     # brown
            'DVAE (rotation)': '#e377c2',     # pink
            'DVAE (blur)': '#7f7f7f',         # gray
            'DVAE (salt_pepper)': '#2f4f4f',  # dark slate gray
            'DVAE (brightness)': '#deb887',   # burlywood
            'DVAE (contrast)': '#6495ed',     # cornflower blue
            # Diffusion
            'Diffusion': '#8B008B',           # dark magenta
            'CLIP-guided Diffusion': '#20B2AA', # light sea green
        }

        # Plot line for each model with specified colors
        for model_name, accuracies in model_results.items():
            color = color_map.get(model_name, '#17becf')  # default cyan if model not in map
            plt.plot(dataset_sizes, accuracies, marker='o', label=model_name, color=color)

        plt.xlabel('Number of Training Samples')
        plt.ylabel('CNN Accuracy (%)')
        plt.title(f'Model Performance vs Dataset Size\n{dataset_type.upper()} Dataset')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        plt.savefig(f'Results/accuracy_vs_size_{dataset_type}.png',
                   bbox_inches='tight', dpi=300)
        plt.close()

    # Print final results
    print("\nFinal Results:")
    for dataset_type in ['balanced', 'semi', 'high']:
        print(f"\n{dataset_type.upper()} Dataset Results:")
        for size in dataset_sizes:
            print(f"\nSize {size}:")
            for name, acc in all_results[dataset_type][size]:
                print(f"{name}: {acc:.2f}%")
