"""
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Generate samples from trained C-VAE

References:
- Sampling from prior mentioned in lecture slide 116
- Generate x by sampling z ~ N(0,I) then decode with condition y
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from cvae import ConditionalVAE
import os

# Create directory for generated data
os.makedirs('generated_data', exist_ok=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load model checkpoint
model_path = 'models/cvae/cvae_latent64_beta4.0.pt'
print(f"Loading model from {model_path}...")
checkpoint = torch.load(model_path, weights_only=False)

input_dim = checkpoint['input_dim']
num_classes = checkpoint['num_classes']
latent_dim = checkpoint['latent_dim']

# Initialize model and load weights
model = ConditionalVAE(input_dim, num_classes, latent_dim=latent_dim).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully\n")

# Load category names
data = torch.load('preprocessed_data/nslkdd_processed.pt', weights_only=False)
category_names = data['category_names']
print(f"Attack categories: {list(category_names)}\n")

# Generate samples for each attack type
num_samples_per_class = 100
print(f"Generating {num_samples_per_class} samples per attack category...")

generated_samples = {}

with torch.no_grad():
    for class_idx in range(num_classes):
        # Create labels for this class
        labels = torch.full((num_samples_per_class,), class_idx, dtype=torch.long).to(device)
        
        # Generate samples
        samples = model.sample(num_samples_per_class, labels, device)
        
        # Store in dict
        class_name = category_names[class_idx]
        generated_samples[class_name] = samples.cpu().numpy()
        
        print(f"  {class_name}: {samples.shape}")

print("\nGeneration complete!")

# Save generated samples
output_path = 'generated_data/cvae_samples.npz'
np.savez(output_path, **generated_samples, category_names=category_names)
print(f"Saved samples to {output_path}")

# Quick visualization - plot first 5 features for each class
print("\nCreating visualization...")
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for idx, (class_name, samples) in enumerate(generated_samples.items()):
    # Plot distributions of first 5 features
    ax = axes[idx]
    for feat_idx in range(5):
        ax.hist(samples[:, feat_idx], bins=20, alpha=0.6, label=f'feat_{feat_idx}')
    
    ax.set_title(f'{class_name}')
    ax.set_xlabel('Feature Value')
    if idx == 0:
        ax.set_ylabel('Count')
    ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = 'plots/cvae/generated_samples_distributions.png'
plt.savefig(plot_path, dpi=200)
print(f"Visualization saved to {plot_path}")

# Print some basic stats
print("\nGenerated sample statistics (first 5 features):")
for class_name, samples in generated_samples.items():
    print(f"\n{class_name}:")
    print(f"  Mean: {samples[:, :5].mean(axis=0)}")
    print(f"  Std: {samples[:, :5].std(axis=0)}")