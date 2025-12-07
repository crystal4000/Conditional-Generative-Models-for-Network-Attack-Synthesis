"""
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Balanced C-GAN Training

Key change: uniform label sampling instead of natural distribution
This forces generator to learn rare classes (R2L, U2R) equally well

References:
- Lecture slides 169-171 on adversarial training
- Slides 178-185 on GAN training procedure
- Professor's suggestion: oversample rare classes during training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
from cgan import ConditionalGAN

# Create directories
os.makedirs('models/cgan', exist_ok=True)
os.makedirs('plots/cgan', exist_ok=True)

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

# Check original class distribution
print("Original training distribution:")
for i in range(num_classes):
    count = (y_train == i).sum().item()
    pct = 100 * count / len(y_train)
    print(f"  Class {i}: {count} samples ({pct:.2f}%)")
print()

# Hyperparameters
noise_dim = 100
batch_size = 256
lr = 0.0002
num_epochs = 100
beta1 = 0.5

print("Hyperparameters:")
print(f"  Noise dim: {noise_dim}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {lr}")
print(f"  Epochs: {num_epochs}\n")

# Create a balanced dataloader
# For each class, we want equal number of samples per epoch
# This is like weighted sampling but simpler for our use case
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = ConditionalGAN(input_dim, num_classes, noise_dim).to(device)
print(f"Generator params: {sum(p.numel() for p in model.generator.parameters())}")
print(f"Discriminator params: {sum(p.numel() for p in model.discriminator.parameters())}\n")

# Optimizers
optimizer_G = optim.Adam(model.generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training history
g_losses = []
d_losses = []

print("Starting balanced training...\n")
print("Key difference: sampling labels uniformly instead of from data distribution")
print("This means U2R and R2L should get equal training signal as Normal\n")

for epoch in range(num_epochs):
    g_loss_epoch = 0
    d_loss_epoch = 0
    
    for batch_idx, (real_data, real_labels) in enumerate(train_loader):
        real_data = real_data.to(device)
        # Still use real_labels for discriminator on real samples
        real_labels_data = real_labels.to(device)
        batch_size_actual = real_data.size(0)
        
        # Real and fake target labels for BCE loss
        real_targets = torch.ones(batch_size_actual, 1).to(device)
        fake_targets = torch.zeros(batch_size_actual, 1).to(device)
        
        # Train Discriminator
        # D should classify real samples correctly AND reject fake samples
        optimizer_D.zero_grad()
        
        # Real samples - use actual labels from data
        d_real = model.discriminator(real_data, real_labels_data)
        d_real_loss = criterion(d_real, real_targets)
        
        # Fake samples - KEY CHANGE: sample labels uniformly
        # instead of using real_labels from batch, we sample random labels
        # this ensures generator practices all classes equally
        # chose uniform sampling to give rare classes equal training signal
        fake_labels = torch.randint(0, num_classes, (batch_size_actual,)).to(device)
        z = torch.randn(batch_size_actual, noise_dim).to(device)
        fake_data = model.generator(z, fake_labels)
        d_fake = model.discriminator(fake_data.detach(), fake_labels)
        d_fake_loss = criterion(d_fake, fake_targets)
        
        # Total discriminator loss from slides 180
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        # G wants D(G(z, y), y) -> 1 (fool discriminator)
        optimizer_G.zero_grad()
        
        # Again, sample labels uniformly
        fake_labels = torch.randint(0, num_classes, (batch_size_actual,)).to(device)
        z = torch.randn(batch_size_actual, noise_dim).to(device)
        fake_data = model.generator(z, fake_labels)
        d_fake = model.discriminator(fake_data, fake_labels)
        
        # Generator loss from slides 179 (non-saturating version)
        g_loss = criterion(d_fake, real_targets)
        g_loss.backward()
        optimizer_G.step()
        
        g_loss_epoch += g_loss.item()
        d_loss_epoch += d_loss.item()
    
    # Average losses
    g_loss_epoch /= len(train_loader)
    d_loss_epoch /= len(train_loader)
    
    g_losses.append(g_loss_epoch)
    d_losses.append(d_loss_epoch)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  G Loss: {g_loss_epoch:.4f}")
        print(f"  D Loss: {d_loss_epoch:.4f}")

print("\nTraining complete!")

# Save model
model_path = 'models/cgan/cgan_balanced_noise100.pt'
torch.save({
    'generator_state_dict': model.generator.state_dict(),
    'discriminator_state_dict': model.discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
    'noise_dim': noise_dim,
    'input_dim': input_dim,
    'num_classes': num_classes,
    'epoch': num_epochs,
    'g_loss': g_losses[-1],
    'd_loss': d_losses[-1]
}, model_path)
print(f"Model saved to {model_path}")

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Balanced C-GAN Training Losses')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = 'plots/cgan/cgan_balanced_noise100_losses.png'
plt.savefig(plot_path, dpi=200)
print(f"Loss curves saved to {plot_path}")