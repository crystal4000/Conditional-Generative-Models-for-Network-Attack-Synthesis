"""
train_smote_gan.py
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Tier 2: Train GAN on SMOTE Data

Train C-GAN on SMOTE-augmented dataset
Now U2R has 500 samples instead of 42, R2L has 5000 instead of 796

Key difference from previous attempts:
SMOTE gave us enough samples for GAN to learn from

References:
- SMOTified-GAN: SMOTE provides initial diversity, GAN refines quality
- Use gradient penalty for stability
- Still use balanced label sampling during GAN training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
from cgan import ConditionalGAN

os.makedirs('models/smote_gan', exist_ok=True)
os.makedirs('plots/smote_gan', exist_ok=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load SMOTE-augmented data
print("Loading SMOTE-augmented data...")
data = torch.load('preprocessed_data/nslkdd_smote_augmented.pt', weights_only=False)
X_train = data['X_train']
y_train = data['y_train']
input_dim = data['input_dim']
num_classes = data['num_classes']

print(f"Input dimension: {input_dim}")
print(f"Number of classes: {num_classes}")
print(f"Training samples: {len(X_train)}")
print(f"Augmentation: {data['augmentation']}\n")

# Show distribution
class_names = ['DoS', 'Normal', 'Probe', 'R2L', 'U2R']
print("Training distribution (after SMOTE):")
for i in range(num_classes):
    count = (y_train == i).sum().item()
    print(f"  {class_names[i]:8s}: {count:6d} samples")
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

# Create dataloader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
# using same architecture as original C-GAN
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

print("Starting training on SMOTE-augmented data...")
print("Now U2R has 500 samples, R2L has 5000 samples")
print("GAN should be able to learn from this expanded dataset\n")

for epoch in range(num_epochs):
    g_loss_epoch = 0
    d_loss_epoch = 0
    
    for batch_idx, (real_data, labels) in enumerate(train_loader):
        real_data = real_data.to(device)
        labels = labels.to(device)
        batch_size_actual = real_data.size(0)
        
        real_targets = torch.ones(batch_size_actual, 1).to(device)
        fake_targets = torch.zeros(batch_size_actual, 1).to(device)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        # Real samples - use actual labels from SMOTE data
        d_real = model.discriminator(real_data, labels)
        d_real_loss = criterion(d_real, real_targets)
        
        # Fake samples - uniform label sampling
        # still important to give equal attention to all classes during training
        fake_labels = torch.randint(0, num_classes, (batch_size_actual,)).to(device)
        z = torch.randn(batch_size_actual, noise_dim).to(device)
        fake_data = model.generator(z, fake_labels)
        d_fake = model.discriminator(fake_data.detach(), fake_labels)
        d_fake_loss = criterion(d_fake, fake_targets)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        fake_labels = torch.randint(0, num_classes, (batch_size_actual,)).to(device)
        z = torch.randn(batch_size_actual, noise_dim).to(device)
        fake_data = model.generator(z, fake_labels)
        d_fake = model.discriminator(fake_data, fake_labels)
        
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
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  G Loss: {g_loss_epoch:.4f}")
        print(f"  D Loss: {d_loss_epoch:.4f}")

print("\nTraining complete!")

# Save model
model_path = 'models/smote_gan/smote_cgan_noise100.pt'
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
    'augmentation': 'SMOTE'
}, model_path)
print(f"Model saved to {model_path}")

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('SMOTE-GAN Training Losses')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = 'plots/smote_gan/training_losses.png'
plt.savefig(plot_path, dpi=200)
print(f"Loss curves saved to {plot_path}")

print("\nNext step: Run evaluate_smote_gan.py")