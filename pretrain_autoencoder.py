"""
pretrain_autoencoder.py
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - BAGAN-GP Phase 1: Autoencoder Pre-training

This is the first step to handle extreme class imbalance.
Pre-train an autoencoder on ALL classes so it learns common patterns
in network traffic. Then we'll use these weights to initialize the GAN.

Key idea from BAGAN-GP paper:
"Train supervised autoencoder first to give generator common knowledge
across all classes before specializing to minority generation"

References:
- BAGAN-GP (2021): autoencoder initialization prevents mode collapse
- Works because encoder learns general traffic patterns from majority classes
- Then can apply that knowledge to reconstruct rare attack types
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os

os.makedirs('models/bagan', exist_ok=True)
os.makedirs('plots/bagan', exist_ok=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load data
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

# Show class distribution
print("Class distribution in training data:")
class_names = ['DoS', 'Normal', 'Probe', 'R2L', 'U2R']
for i in range(num_classes):
    count = (y_train == i).sum().item()
    print(f"  {class_names[i]:8s}: {count:6d} samples")
print()

# Supervised Autoencoder Architecture
# encoder learns to compress data + class info into latent space
# decoder reconstructs from latent + class
class SupervisedAutoencoder(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim=128):
        super(SupervisedAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Encoder: [input + class] -> latent
        # learns common patterns across all traffic types
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder: [latent + class] -> reconstructed input
        # learns to rebuild traffic features from compressed representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, input_dim)
            # no activation on output for continuous features
        )
        
        # Optional classifier on latent space
        # helps ensure different classes map to different latent regions
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes)
        )
    
    def encode(self, x, labels):
        # convert labels to one-hot
        labels_onehot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        # concatenate input and class label
        x_with_label = torch.cat([x, labels_onehot], dim=1)
        return self.encoder(x_with_label)
    
    def decode(self, z, labels):
        # convert labels to one-hot
        labels_onehot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        # concatenate latent and class label
        z_with_label = torch.cat([z, labels_onehot], dim=1)
        return self.decoder(z_with_label)
    
    def forward(self, x, labels):
        z = self.encode(x, labels)
        reconstructed = self.decode(z, labels)
        class_pred = self.classifier(z)
        return reconstructed, class_pred, z

# Hyperparameters
latent_dim = 128
batch_size = 256
lr = 0.0002
num_epochs = 100
# weight for reconstruction vs classification
# reconstruction is more important for autoencoder initialization
recon_weight = 1.0
class_weight = 0.1

print("Hyperparameters:")
print(f"  Latent dim: {latent_dim}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {lr}")
print(f"  Epochs: {num_epochs}")
print(f"  Reconstruction weight: {recon_weight}")
print(f"  Classification weight: {class_weight}\n")

# Create dataloaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = SupervisedAutoencoder(input_dim, num_classes, latent_dim).to(device)
print(f"Autoencoder parameters: {sum(p.numel() for p in model.parameters())}\n")

# Optimizer and loss functions
optimizer = optim.Adam(model.parameters(), lr=lr)
recon_criterion = nn.MSELoss()  # reconstruction loss
class_criterion = nn.CrossEntropyLoss()  # classification loss

# Training history
train_losses = []
val_losses = []
recon_losses = []
class_losses = []

print("Starting autoencoder pre-training...")


for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss_epoch = 0
    recon_loss_epoch = 0
    class_loss_epoch = 0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, class_pred, latent = model(data, labels)
        
        # Reconstruction loss - how well can we rebuild the input?
        recon_loss = recon_criterion(reconstructed, data)
        
        # Classification loss - can we predict class from latent?
        # this ensures different classes occupy different latent regions
        class_loss = class_criterion(class_pred, labels)
        
        # Combined loss
        total_loss = recon_weight * recon_loss + class_weight * class_loss
        
        total_loss.backward()
        optimizer.step()
        
        train_loss_epoch += total_loss.item()
        recon_loss_epoch += recon_loss.item()
        class_loss_epoch += class_loss.item()
    
    train_loss_epoch /= len(train_loader)
    recon_loss_epoch /= len(train_loader)
    class_loss_epoch /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss_epoch = 0
    
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            reconstructed, class_pred, latent = model(data, labels)
            recon_loss = recon_criterion(reconstructed, data)
            class_loss = class_criterion(class_pred, labels)
            total_loss = recon_weight * recon_loss + class_weight * class_loss
            
            val_loss_epoch += total_loss.item()
    
    val_loss_epoch /= len(val_loader)
    
    train_losses.append(train_loss_epoch)
    val_losses.append(val_loss_epoch)
    recon_losses.append(recon_loss_epoch)
    class_losses.append(class_loss_epoch)
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss_epoch:.4f}")
        print(f"  Val Loss: {val_loss_epoch:.4f}")
        print(f"  Recon: {recon_loss_epoch:.4f}, Class: {class_loss_epoch:.4f}")

print("\nPre-training complete!")

# Save model
model_path = 'models/bagan/pretrained_autoencoder.pt'
torch.save({
    'encoder_state_dict': model.encoder.state_dict(),
    'decoder_state_dict': model.decoder.state_dict(),
    'classifier_state_dict': model.classifier.state_dict(),
    'latent_dim': latent_dim,
    'input_dim': input_dim,
    'num_classes': num_classes,
    'epoch': num_epochs,
    'train_loss': train_losses[-1],
    'val_loss': val_losses[-1]
}, model_path)
print(f"Pretrained autoencoder saved to {model_path}")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Overall loss
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Autoencoder Training Loss')
ax1.legend()
ax1.grid(alpha=0.3)

# Component losses
ax2.plot(recon_losses, label='Reconstruction Loss')
ax2.plot(class_losses, label='Classification Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Components')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plot_path = 'plots/bagan/autoencoder_pretraining_losses.png'
plt.savefig(plot_path, dpi=200)
print(f"Loss curves saved to {plot_path}")

