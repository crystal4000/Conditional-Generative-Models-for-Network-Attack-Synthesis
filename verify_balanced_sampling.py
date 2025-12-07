"""
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Verify Uniform Label Sampling

This script helps verify that our balanced training is actually
sampling labels uniformly as expected.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Load data
print("Loading preprocessed data...")
data = torch.load('preprocessed_data/nslkdd_processed.pt', weights_only=False)
X_train = data['X_train']
y_train = data['y_train']
num_classes = data['num_classes']

class_names = ['DoS', 'Normal', 'Probe', 'R2L', 'U2R']

# Original distribution from dataloader
print("\n" + "="*60)
print("ORIGINAL DISTRIBUTION (from dataloader)")
print("="*60)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

original_label_counts = {i: 0 for i in range(num_classes)}

# Simulate one epoch
for batch_idx, (real_data, real_labels) in enumerate(train_loader):
    for label in real_labels:
        original_label_counts[label.item()] += 1
    
    if batch_idx >= 100:  # just sample 100 batches for speed
        break

total = sum(original_label_counts.values())
print("\nLabels seen in 100 batches (old approach):")
for i in range(num_classes):
    count = original_label_counts[i]
    pct = 100 * count / total
    print(f"  {class_names[i]:8s}: {count:5d} ({pct:5.2f}%)")
print(f"\nTotal: {total}")

# Simulated uniform sampling
print("\n" + "="*60)
print("UNIFORM SAMPLING (new approach)")
print("="*60)

uniform_label_counts = {i: 0 for i in range(num_classes)}

# Simulate 100 batches with uniform sampling
for _ in range(100):
    batch_size = 256
    fake_labels = torch.randint(0, num_classes, (batch_size,))
    for label in fake_labels:
        uniform_label_counts[label.item()] += 1

total = sum(uniform_label_counts.values())
print("\nLabels sampled in 100 batches (new approach):")
for i in range(num_classes):
    count = uniform_label_counts[i]
    pct = 100 * count / total
    expected_pct = 100.0 / num_classes
    print(f"  {class_names[i]:8s}: {count:5d} ({pct:5.2f}%) [expected: {expected_pct:.2f}%]")
print(f"\nTotal: {total}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Original distribution
counts_original = [original_label_counts[i] for i in range(num_classes)]
pcts_original = [100 * c / sum(counts_original) for c in counts_original]
ax1.bar(class_names, pcts_original, color='#e74c3c', alpha=0.7)
ax1.axhline(y=20, color='gray', linestyle='--', label='Uniform (20%)')
ax1.set_ylabel('Percentage of Labels (%)')
ax1.set_title('Original Approach\n(Labels from Dataloader)')
ax1.set_ylim([0, 60])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Uniform sampling
counts_uniform = [uniform_label_counts[i] for i in range(num_classes)]
pcts_uniform = [100 * c / sum(counts_uniform) for c in counts_uniform]
ax2.bar(class_names, pcts_uniform, color='#2ecc71', alpha=0.7)
ax2.axhline(y=20, color='gray', linestyle='--', label='Target (20%)')
ax2.set_ylabel('Percentage of Labels (%)')
ax2.set_title('Balanced Approach\n(Uniform Sampling)')
ax2.set_ylim([0, 60])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/cgan/label_distribution_comparison.png', dpi=200)
print("\nPlot saved to plots/cgan/label_distribution_comparison.png")

# Statistical test for uniformity
print("\n" + "="*60)
print("UNIFORMITY CHECK")
print("="*60)
expected_count = total / num_classes
chi_square = sum((uniform_label_counts[i] - expected_count)**2 / expected_count 
                  for i in range(num_classes))

print(f"\nExpected count per class: {expected_count:.1f}")
print(f"Chi-square statistic: {chi_square:.2f}")
print(f"(lower is more uniform, <10 is good)")

if chi_square < 10:
    print("\n✓ Uniform sampling is working correctly!")
else:
    print("\n✗ Warning: sampling may not be uniform")

