"""
train_cgan.py
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - C-GAN Training

References:
- Lecture slide 169-171 on adversarial training
- Discriminator and Generator play a zero-sum game
- Proposal: lr=0.0002, batch_size=128-256, 100 epochs
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

# Hyperparameters from proposal
noise_dim = 100
batch_size = 256
lr = 0.0002  # standard GAN learning rate
num_epochs = 300
beta1 = 0.5  # Adam beta1 for GANs

print("Hyperparameters:")
print(f"  Noise dim: {noise_dim}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {lr}")
print(f"  Epochs: {num_epochs}\n")

# Create dataloader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = ConditionalGAN(input_dim, num_classes, noise_dim).to(device)
print(f"Generator params: {sum(p.numel() for p in model.generator.parameters())}")
print(f"Discriminator params: {sum(p.numel() for p in model.discriminator.parameters())}\n")

# Optimizers - separate for G and D
optimizer_G = optim.Adam(model.generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Loss function - binary cross entropy
criterion = nn.BCELoss()

# Training history
g_losses = []
d_losses = []

print("Starting training...\n")

for epoch in range(num_epochs):
    g_loss_epoch = 0
    d_loss_epoch = 0
    
    for batch_idx, (real_data, labels) in enumerate(train_loader):
        real_data = real_data.to(device)
        labels = labels.to(device)
        batch_size_actual = real_data.size(0)
        
        # Real and fake labels
        real_labels = torch.ones(batch_size_actual, 1).to(device)
        fake_labels = torch.zeros(batch_size_actual, 1).to(device)
        
        # Train Discriminator
        # want D(real, y) -> 1 and D(G(z, y), y) -> 0
        optimizer_D.zero_grad()
        
        # Real samples
        d_real = model.discriminator(real_data, labels)
        d_real_loss = criterion(d_real, real_labels)
        
        # Fake samples
        z = torch.randn(batch_size_actual, noise_dim).to(device)
        fake_data = model.generator(z, labels)
        d_fake = model.discriminator(fake_data.detach(), labels)
        d_fake_loss = criterion(d_fake, fake_labels)
        
        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        # want D(G(z, y), y) -> 1 (fool the discriminator)
        optimizer_G.zero_grad()
        
        z = torch.randn(batch_size_actual, noise_dim).to(device)
        fake_data = model.generator(z, labels)
        d_fake = model.discriminator(fake_data, labels)
        
        # Generator tries to maximize D(G(z))
        g_loss = criterion(d_fake, real_labels)
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
model_path = f'models/cgan/cgan_noise{noise_dim}_epochs{num_epochs}.pt'
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
plt.title('GAN Training Losses')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = f'plots/cgan/cgan_noise{noise_dim}_epochs{num_epochs}_losses.png'
plt.savefig(plot_path, dpi=200)
print(f"Loss curves saved to {plot_path}")