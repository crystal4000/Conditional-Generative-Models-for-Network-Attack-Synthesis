"""
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Sample from Balanced C-GAN

Generate synthetic attack samples with balanced class representation
"""

import torch
import numpy as np
import os
from cgan import ConditionalGAN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load model
print("Loading balanced C-GAN model...")
checkpoint = torch.load('models/cgan/cgan_balanced_noise100.pt', 
                       map_location=device, weights_only=False)

input_dim = checkpoint['input_dim']
num_classes = checkpoint['num_classes']
noise_dim = checkpoint['noise_dim']

model = ConditionalGAN(input_dim, num_classes, noise_dim).to(device)
model.generator.load_state_dict(checkpoint['generator_state_dict'])
model.generator.eval()

print(f"Model info:")
print(f"  Input dim: {input_dim}")
print(f"  Classes: {num_classes}")
print(f"  Noise dim: {noise_dim}")
print(f"  Trained for {checkpoint['epoch']} epochs\n")

# Generate balanced samples
# equal samples per class to demonstrate balanced generation
samples_per_class = 1000
class_names = ['DoS', 'Normal', 'Probe', 'R2L', 'U2R']

print(f"Generating {samples_per_class} samples per class...")

all_samples = []
all_labels = []

with torch.no_grad():
    for class_idx in range(num_classes):
        print(f"  Generating {class_names[class_idx]}...")
        
        # Create labels for this class
        labels = torch.full((samples_per_class,), class_idx, dtype=torch.long).to(device)
        
        # Sample noise
        z = torch.randn(samples_per_class, noise_dim).to(device)
        
        # Generate samples
        samples = model.generator(z, labels)
        
        all_samples.append(samples.cpu().numpy())
        all_labels.extend([class_idx] * samples_per_class)

# Combine all samples
all_samples = np.vstack(all_samples)
all_labels = np.array(all_labels)

print(f"\nGenerated {len(all_samples)} total samples")
print(f"Shape: {all_samples.shape}")

# Save samples
os.makedirs('generated_data', exist_ok=True)
output_path = 'generated_data/cgan_balanced_samples.npz'

np.savez(output_path,
         samples=all_samples,
         labels=all_labels,
         class_names=class_names)

print(f"\nSamples saved to {output_path}")

# Show distribution
print("\nGenerated sample distribution:")
for i, name in enumerate(class_names):
    count = (all_labels == i).sum()
    print(f"  {name}: {count} samples")