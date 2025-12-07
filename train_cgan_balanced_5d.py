"""
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Balanced C-GAN with 5:1 D Updates

Key changes:
1. Uniform label sampling (addresses class imbalance)
2. 5 discriminator updates per 1 generator update (stabilizes training)

References:
- HW3 Problem 1.5: discriminator update frequency affects stability
- When D is too strong, G can't learn (high G loss ~40 indicates this)
- More D updates = stronger D signal for G to learn from
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
from cgan import ConditionalGAN

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

# Hyperparameters
noise_dim = 100
batch_size = 256
lr = 0.0002
num_epochs = 150  # increased from 100
beta1 = 0.5
d_updates = 5  # discriminator updates per generator update

print("Hyperparameters:")
print(f"  Noise dim: {noise_dim}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {lr}")
print(f"  Epochs: {num_epochs}")
print(f"  D updates per G update: {d_updates}\n")

# Create dataloader
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

print("Starting balanced training with 5:1 D:G update ratio...\n")
print("Stabilizing training and preventing discriminator from becoming too strong\n")

for epoch in range(num_epochs):
    g_loss_epoch = 0
    d_loss_epoch = 0
    
    for batch_idx, (real_data, real_labels) in enumerate(train_loader):
        real_data = real_data.to(device)
        real_labels_data = real_labels.to(device)
        batch_size_actual = real_data.size(0)
        
        real_targets = torch.ones(batch_size_actual, 1).to(device)
        fake_targets = torch.zeros(batch_size_actual, 1).to(device)
        
        # Train Discriminator 5 times
        # from HW3: more D updates helps when G is struggling
        d_loss_batch = 0
        for _ in range(d_updates):
            optimizer_D.zero_grad()
            
            # Real samples
            d_real = model.discriminator(real_data, real_labels_data)
            d_real_loss = criterion(d_real, real_targets)
            
            # Fake samples with uniform label sampling
            fake_labels = torch.randint(0, num_classes, (batch_size_actual,)).to(device)
            z = torch.randn(batch_size_actual, noise_dim).to(device)
            fake_data = model.generator(z, fake_labels)
            d_fake = model.discriminator(fake_data.detach(), fake_labels)
            d_fake_loss = criterion(d_fake, fake_targets)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()
            
            d_loss_batch += d_loss.item()
        
        d_loss_batch /= d_updates
        
        # Train Generator once
        optimizer_G.zero_grad()
        
        fake_labels = torch.randint(0, num_classes, (batch_size_actual,)).to(device)
        z = torch.randn(batch_size_actual, noise_dim).to(device)
        fake_data = model.generator(z, fake_labels)
        d_fake = model.discriminator(fake_data, fake_labels)
        
        g_loss = criterion(d_fake, real_targets)
        g_loss.backward()
        optimizer_G.step()
        
        g_loss_epoch += g_loss.item()
        d_loss_epoch += d_loss_batch
    
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
model_path = 'models/cgan/cgan_balanced_5d_noise100.pt'
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
    'd_loss': d_losses[-1],
    'd_updates': d_updates
}, model_path)
print(f"Model saved to {model_path}")

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Balanced C-GAN Training (5:1 D:G Updates)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = 'plots/cgan/cgan_balanced_5d_noise100_losses.png'
plt.savefig(plot_path, dpi=200)
print(f"Loss curves saved to {plot_path}")