"""
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Generate samples from trained C-GAN

References:
- Sample from prior z ~ N(0,I), then G(z, y) generates conditioned samples
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from cgan import ConditionalGAN
import os

os.makedirs('generated_data', exist_ok=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load model
model_path = 'models/cgan/cgan_noise100.pt'
print(f"Loading model from {model_path}...")
checkpoint = torch.load(model_path, weights_only=False)

input_dim = checkpoint['input_dim']
num_classes = checkpoint['num_classes']
noise_dim = checkpoint['noise_dim']

model = ConditionalGAN(input_dim, num_classes, noise_dim).to(device)
model.generator.load_state_dict(checkpoint['generator_state_dict'])
model.generator.eval()
print("Model loaded\n")

# Load category names
data = torch.load('preprocessed_data/nslkdd_processed.pt', weights_only=False)
category_names = data['category_names']
print(f"Categories: {list(category_names)}\n")

# Generate samples
num_samples_per_class = 100
print(f"Generating {num_samples_per_class} samples per category...")

generated_samples = {}

with torch.no_grad():
    for class_idx in range(num_classes):
        labels = torch.full((num_samples_per_class,), class_idx, dtype=torch.long).to(device)
        samples = model.generate(num_samples_per_class, labels, device)
        
        class_name = category_names[class_idx]
        generated_samples[class_name] = samples.cpu().numpy()
        print(f"  {class_name}: {samples.shape}")

print("\nGeneration complete!")

# Save samples
output_path = 'generated_data/cgan_samples.npz'
np.savez(output_path, **generated_samples, category_names=category_names)
print(f"Saved to {output_path}")

# Visualization
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for idx, (class_name, samples) in enumerate(generated_samples.items()):
    ax = axes[idx]
    for feat_idx in range(5):
        ax.hist(samples[:, feat_idx], bins=20, alpha=0.6, label=f'feat_{feat_idx}')
    
    ax.set_title(f'{class_name}')
    ax.set_xlabel('Feature Value')
    if idx == 0:
        ax.set_ylabel('Count')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/cgan/generated_samples_distributions.png', dpi=200)
print("Visualization saved")

print("\nSample statistics (first 5 features):")
for class_name, samples in generated_samples.items():
    print(f"\n{class_name}:")
    print(f"  Mean: {samples[:, :5].mean(axis=0)}")
    print(f"  Std: {samples[:, :5].std(axis=0)}")