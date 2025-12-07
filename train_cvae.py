"""
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - C-VAE Training

References:
- Lecture slide 116 for VAE training loop pseudocode
- Proposal: Adam optimizer with lr=0.001, batch_size=128-256, 50-100 epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
from cvae import ConditionalVAE, vae_loss

# Create directories if they don't exist
os.makedirs('models/cvae', exist_ok=True)
os.makedirs('plots/cvae', exist_ok=True)

# Device setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load preprocessed data
print("Loading preprocessed data...")
data = torch.load('preprocessed_data/nslkdd_processed.pt', weights_only=False)
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
input_dim = data['input_dim']
num_classes = data['num_classes']

print(f"Input dimension: {input_dim}")
print(f"Number of classes: {num_classes}")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}\n")

# Hyperparameters from proposal
latent_dim = 64  # testing with 64, can also try 32 and 128
batch_size = 256
learning_rate = 0.001
num_epochs = 50
beta = 4.0  # stronger KL penalty to improve disentanglement (slide 141)

print("Hyperparameters:")
print(f"  Latent dim: {latent_dim}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")
print(f"  Epochs: {num_epochs}")
print(f"  Beta: {beta}\n")

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = ConditionalVAE(input_dim, num_classes, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters\n")

# Training history
train_losses = []
val_losses = []
train_recon_losses = []
train_kl_losses = []

print("Starting training...\n")

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_recon = 0
    train_kl = 0
    
    for batch_idx, (x, labels) in enumerate(train_loader):
        x = x.to(device)
        labels = labels.to(device)
        
        # Forward pass
        x_recon, mu, logvar = model(x, labels)
        
        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar, beta=beta)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_recon += recon_loss.item()
        train_kl += kl_loss.item()
    
    # Average losses
    train_loss /= len(train_loader.dataset)
    train_recon /= len(train_loader.dataset)
    train_kl /= len(train_loader.dataset)
    
    train_losses.append(train_loss)
    train_recon_losses.append(train_recon)
    train_kl_losses.append(train_kl)
    
    # Validation phase
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for x, labels in val_loader:
            x = x.to(device)
            labels = labels.to(device)
            
            x_recon, mu, logvar = model(x, labels)
            loss, _, _ = vae_loss(x_recon, x, mu, logvar, beta=beta)
            val_loss += loss.item()
    
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    
    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
        print(f"  Val Loss: {val_loss:.4f}")

print("\nTraining complete!")

# Save model
model_path = f'models/cvae/cvae_latent{latent_dim}_beta{beta}.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'latent_dim': latent_dim,
    'input_dim': input_dim,
    'num_classes': num_classes,
    'beta': beta,
    'epoch': num_epochs,
    'train_loss': train_losses[-1],
    'val_loss': val_losses[-1]
}, model_path)
print(f"Model saved to {model_path}")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(val_losses, label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Total Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Reconstruction vs KL
axes[1].plot(train_recon_losses, label='Reconstruction Loss')
axes[1].plot(train_kl_losses, label='KL Divergence')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Loss Components')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plot_path = f'plots/cvae/cvae_latent{latent_dim}_beta{beta}_losses.png'
plt.savefig(plot_path, dpi=200)
print(f"Loss curves saved to {plot_path}")