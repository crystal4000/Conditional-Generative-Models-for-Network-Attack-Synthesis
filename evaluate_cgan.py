"""
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Evaluate C-GAN Generated Samples

References:
- Proposal: >80% classification accuracy target
"""

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\nEvaluating C-GAN Generated Samples\n")

# Load real data
data = torch.load('preprocessed_data/nslkdd_processed.pt', weights_only=False)
X_train = data['X_train'].numpy()
y_train = data['y_train'].numpy()
X_test = data['X_test'].numpy()
y_test = data['y_test'].numpy()
category_names = data['category_names']

print(f"Categories: {list(category_names)}\n")

# Train classifier
print("Training classifier on real data...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Baseline
y_pred_real = clf.predict(X_test)
acc_real = accuracy_score(y_test, y_pred_real)
print(f"Baseline (real test): {acc_real:.4f}\n")

# Load C-GAN samples
print("Loading C-GAN generated samples...")
gen_data = np.load('generated_data/cgan_samples.npz')

X_gen_list = []
y_gen_list = []

for i, cat in enumerate(category_names):
    samples = gen_data[cat]
    X_gen_list.append(samples)
    y_gen_list.append(np.full(len(samples), i))

X_gen = np.vstack(X_gen_list)
y_gen = np.concatenate(y_gen_list)

print(f"Generated samples: {len(X_gen)}\n")

# Evaluate
y_pred_gen = clf.predict(X_gen)
acc_gen = accuracy_score(y_gen, y_pred_gen)

print(f"C-GAN accuracy: {acc_gen:.4f}")
if acc_gen >= 0.80:
    print("SUCCESS: Meets 80% threshold!")
else:
    print(f"Below 80% threshold (gap: {0.80 - acc_gen:.4f})")

print("\nPer-class accuracy:")
for i, cat in enumerate(category_names):
    mask = y_gen == i
    acc = accuracy_score(y_gen[mask], y_pred_gen[mask])
    print(f"  {cat}: {acc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_gen, y_pred_gen)
print("\nConfusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=category_names, yticklabels=category_names, ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix - C-GAN Generated Samples')
plt.tight_layout()
plt.savefig('plots/cgan/confusion_matrix_generated.png', dpi=200)
print("\nSaved confusion matrix to plots/cgan/confusion_matrix_generated.png")

print("\n=== Classification Report ===")
print(classification_report(y_gen, y_pred_gen, target_names=category_names))

print("\n=== Summary ===")
print(f"Real data accuracy: {acc_real:.4f}")
print(f"C-GAN accuracy: {acc_gen:.4f}")
print(f"Difference: {acc_real - acc_gen:.4f}")