"""
smote_augmentation.py
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Tier 2: SMOTE Preprocessing

Apply SMOTE to expand rare classes BEFORE training GAN
This gives the GAN enough samples to learn from

From research: "GAN has not been used for small datasets because
there's not enough information for generator to learn"

Strategy:
- Expand U2R: 42 → 500 samples
- Expand R2L: 796 → 5000 samples
- Keep DoS, Normal, Probe original
- Then train GAN on this expanded dataset

References:
- SMOTified-GAN (2022): SMOTE + GAN outperforms either alone
- SMOTE creates interpolated samples as starting point
- GAN then learns to transform them into more realistic distributions
"""

import torch
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os

os.makedirs('preprocessed_data', exist_ok=True)
os.makedirs('plots/smote', exist_ok=True)

print("SMOTE Augmentation for Rare Classes\n")

# Load original data
print("Loading original preprocessed data...")
data = torch.load('preprocessed_data/nslkdd_processed.pt', weights_only=False)
X_train = data['X_train'].numpy()
y_train = data['y_train'].numpy()
X_val = data['X_val'].numpy()
y_val = data['y_val'].numpy()
X_test = data['X_test']
y_test = data['y_test']
input_dim = data['input_dim']
num_classes = data['num_classes']

class_names = ['DoS', 'Normal', 'Probe', 'R2L', 'U2R']

print("\nOriginal training distribution:")
for i in range(num_classes):
    count = (y_train == i).sum()
    print(f"  {class_names[i]:8s}: {count:6d} samples")

# Define target samples for each class
# expand rare classes while keeping majority classes
target_samples = {
    0: (y_train == 0).sum(),  # DoS: keep original (36,741)
    1: (y_train == 1).sum(),  # Normal: keep original (53,874)
    2: (y_train == 2).sum(),  # Probe: keep original (9,325)
    3: 5000,  # R2L: expand from 796 to 5000
    4: 500    # U2R: expand from 42 to 500
}

print("\nTarget distribution after SMOTE:")
for i in range(num_classes):
    print(f"  {class_names[i]:8s}: {target_samples[i]:6d} samples")

# Apply SMOTE
# k_neighbors must be less than minority class size
# for U2R (42 samples), we can use max k=5
print("\nApplying SMOTE...")
print("Using k_neighbors=5 (limited by U2R class with 42 samples)")

smote = SMOTE(
    sampling_strategy=target_samples,
    k_neighbors=5,  # must be <= smallest class size - 1
    random_state=42
)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nSMOTE complete!")
print(f"Original samples: {len(X_train)}")
print(f"After SMOTE: {len(X_train_resampled)}")

print("\nNew distribution:")
for i in range(num_classes):
    count = (y_train_resampled == i).sum()
    original = (y_train == i).sum()
    increase = count - original
    print(f"  {class_names[i]:8s}: {count:6d} (+{increase} synthetic)")

# Convert back to tensors
X_train_resampled = torch.FloatTensor(X_train_resampled)
y_train_resampled = torch.LongTensor(y_train_resampled)

# Save augmented dataset
# keep validation and test sets unchanged
print("\nSaving SMOTE-augmented dataset...")
torch.save({
    'X_train': X_train_resampled,
    'y_train': y_train_resampled,
    'X_val': X_val,  # original validation
    'y_val': y_val,
    'X_test': X_test,  # original test
    'y_test': y_test,
    'input_dim': input_dim,
    'num_classes': num_classes,
    'augmentation': 'SMOTE',
    'target_samples': target_samples
}, 'preprocessed_data/nslkdd_smote_augmented.pt')
print("Saved to preprocessed_data/nslkdd_smote_augmented.pt")

# Visualize distribution change
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Original distribution
original_counts = [(y_train == i).sum() for i in range(num_classes)]
ax1.bar(class_names, original_counts, color='#e74c3c', alpha=0.7)
ax1.set_ylabel('Sample Count')
ax1.set_title('Original Training Distribution')
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)

# After SMOTE
smote_counts = [(y_train_resampled == i).sum().item() for i in range(num_classes)]
colors = ['#e74c3c' if i < 3 else '#2ecc71' for i in range(num_classes)]
ax2.bar(class_names, smote_counts, color=colors, alpha=0.7)
ax2.set_ylabel('Sample Count')
ax2.set_title('After SMOTE Augmentation')
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/smote/distribution_comparison.png', dpi=200)
print("Distribution plot saved to plots/smote/distribution_comparison.png")


print("SMOTE AUGMENTATION COMPLETE")

