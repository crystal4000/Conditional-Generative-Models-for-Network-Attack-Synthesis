"""
train_bagan_gp.py
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - BAGAN-GP Training

Phase 2 of handling extreme class imbalance:
1. Initialize GAN from pretrained autoencoder (Phase 1)
2. Train with gradient penalty for stability
3. Use balanced label sampling (uniform distribution)
4. 5:1 discriminator:generator update ratio

Key difference from previous attempts:
Generator starts with knowledge from 67k Normal samples,
then learns to apply that to generate rare U2R/R2L attacks

References:
- BAGAN-GP paper: GP helps when discriminator has limited minority samples
- λ=10 is standard for gradient penalty
- Balanced sampling ensures equal training signal for all classes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
from bagan_gp import BAGAN_GP, compute_gradient_penalty

os.makedirs('models/bagan', exist_ok=True)
os.makedirs('plots/bagan', exist_ok=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load data
print("Loading preprocessed data...")
data = torch.load('preprocessed_data/nslkdd_processed.pt', weights_only=False)
X_train = data['X_train']
y_train = data['y_train']
input_dim = data['input_dim']
num_classes = data['num_classes']

print(f"Input dimension: {input_dim}")
print(f"Number of classes: {num_classes}")
print(f"Training samples: {len(X_train)}\n")

# Hyperparameters
noise_dim = 100
latent_dim = 128
batch_size = 256
lr = 0.0001  # slightly lower than standard for stability
num_epochs = 150
d_updates = 5  # discriminator updates per generator update
lambda_gp = 10.0  # gradient penalty weight (standard value)

print("Hyperparameters:")
print(f"  Noise dim: {noise_dim}")
print(f"  Latent dim: {latent_dim}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {lr}")
print(f"  Epochs: {num_epochs}")
print(f"  D updates per G update: {d_updates}")
print(f"  Gradient penalty λ: {lambda_gp}\n")

# Create dataloader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = BAGAN_GP(input_dim, num_classes, noise_dim, latent_dim).to(device)

# Load pretrained autoencoder weights
# this is the key step that gives us common knowledge across classes
model.load_pretrained_weights('models/bagan/pretrained_autoencoder.pt')

print(f"Generator params: {sum(p.numel() for p in model.generator.parameters())}")
print(f"Discriminator params: {sum(p.numel() for p in model.discriminator.parameters())}\n")

# Optimizers
# using Adam with beta1=0.5 as recommended for GANs
optimizer_G = optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training history
g_losses = []
d_losses = []
gp_values = []

print("Starting BAGAN-GP training...")
print("Key improvements over previous attempts:")
print("  1. Generator initialized with autoencoder knowledge")
print("  2. Gradient penalty prevents discriminator saturation")
print("  3. Uniform label sampling (not data distribution)")
print("  4. 5:1 D:G ratio for stability\n")

for epoch in range(num_epochs):
    g_loss_epoch = 0
    d_loss_epoch = 0
    gp_epoch = 0
    
    for batch_idx, (real_data, real_labels) in enumerate(train_loader):
        real_data = real_data.to(device)
        real_labels_data = real_labels.to(device)
        batch_size_actual = real_data.size(0)
        
        real_targets = torch.ones(batch_size_actual, 1).to(device)
        fake_targets = torch.zeros(batch_size_actual, 1).to(device)
        
        # Train Discriminator 5 times
        # more updates help D learn from limited minority samples
        d_loss_batch = 0
        gp_batch = 0
        
        for _ in range(d_updates):
            optimizer_D.zero_grad()
            
            # Real samples
            d_real = model.discriminator(real_data, real_labels_data)
            d_real_loss = criterion(d_real, real_targets)
            
            # Fake samples with UNIFORM label sampling
            # this is critical - gives U2R and R2L equal training signal
            fake_labels = torch.randint(0, num_classes, (batch_size_actual,)).to(device)
            z = torch.randn(batch_size_actual, noise_dim).to(device)
            fake_data = model.generator(z, fake_labels)
            d_fake = model.discriminator(fake_data.detach(), fake_labels)
            d_fake_loss = criterion(d_fake, fake_targets)
            
            # Gradient penalty
            # prevents D from becoming overconfident on rare classes
            gp = compute_gradient_penalty(
                model.discriminator, 
                real_data, 
                fake_data.detach(), 
                fake_labels, 
                device
            )
            
            # Total discriminator loss with gradient penalty
            d_loss = d_real_loss + d_fake_loss + lambda_gp * gp
            d_loss.backward()
            optimizer_D.step()
            
            d_loss_batch += d_loss.item()
            gp_batch += gp.item()
        
        d_loss_batch /= d_updates
        gp_batch /= d_updates
        
        # Train Generator once
        optimizer_G.zero_grad()
        
        # Uniform label sampling for generator too
        fake_labels = torch.randint(0, num_classes, (batch_size_actual,)).to(device)
        z = torch.randn(batch_size_actual, noise_dim).to(device)
        fake_data = model.generator(z, fake_labels)
        d_fake = model.discriminator(fake_data, fake_labels)
        
        # Generator wants to fool discriminator
        g_loss = criterion(d_fake, real_targets)
        g_loss.backward()
        optimizer_G.step()
        
        g_loss_epoch += g_loss.item()
        d_loss_epoch += d_loss_batch
        gp_epoch += gp_batch
    
    # Average losses
    g_loss_epoch /= len(train_loader)
    d_loss_epoch /= len(train_loader)
    gp_epoch /= len(train_loader)
    
    g_losses.append(g_loss_epoch)
    d_losses.append(d_loss_epoch)
    gp_values.append(gp_epoch)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  G Loss: {g_loss_epoch:.4f}")
        print(f"  D Loss: {d_loss_epoch:.4f}")
        print(f"  GP: {gp_epoch:.4f}")

print("\nTraining complete!")

# Save model
model_path = 'models/bagan/bagan_gp_final.pt'
torch.save({
    'generator_state_dict': model.generator.state_dict(),
    'discriminator_state_dict': model.discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
    'noise_dim': noise_dim,
    'latent_dim': latent_dim,
    'input_dim': input_dim,
    'num_classes': num_classes,
    'epoch': num_epochs,
    'g_loss': g_losses[-1],
    'd_loss': d_losses[-1],
    'lambda_gp': lambda_gp
}, model_path)
print(f"Model saved to {model_path}")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Generator and Discriminator losses
ax1.plot(g_losses, label='Generator Loss', alpha=0.8)
ax1.plot(d_losses, label='Discriminator Loss', alpha=0.8)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('BAGAN-GP Training Losses')
ax1.legend()
ax1.grid(alpha=0.3)

# Gradient penalty over time
ax2.plot(gp_values, label='Gradient Penalty', color='green', alpha=0.8)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Gradient Penalty')
ax2.set_title('Gradient Penalty (stabilizes discriminator)')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plot_path = 'plots/bagan/bagan_gp_training_losses.png'
plt.savefig(plot_path, dpi=200)
print(f"Loss curves saved to {plot_path}")

