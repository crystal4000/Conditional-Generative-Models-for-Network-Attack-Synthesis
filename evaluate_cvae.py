"""
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Evaluate C-VAE Generated Samples

References:
- Proposal: Generated attacks should be classifiable with >80% accuracy
- This tests if our conditional generation actually produces distinguishable attack types
"""

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\nEvaluating C-VAE Generated Samples\n")

# Load real data
print("Loading real data...")
data = torch.load('preprocessed_data/nslkdd_processed.pt', weights_only=False)
X_train = data['X_train'].numpy()
y_train = data['y_train'].numpy()
X_test = data['X_test'].numpy()
y_test = data['y_test'].numpy()
category_names = data['category_names']

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Categories: {list(category_names)}\n")

# Train a classifier on real data
print("Training classifier on real data...")
# using Random Forest since it works well for tabular data
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
print("Classifier trained\n")

# Evaluate on real test data first (baseline)
print("=== Baseline: Real Test Data ===")
y_pred_real = clf.predict(X_test)
acc_real = accuracy_score(y_test, y_pred_real)
print(f"Accuracy on real test data: {acc_real:.4f}\n")

print("Per-class accuracy:")
for i, cat in enumerate(category_names):
    mask = y_test == i
    if mask.sum() > 0:
        acc = accuracy_score(y_test[mask], y_pred_real[mask])
        print(f"  {cat}: {acc:.4f} ({mask.sum()} samples)")

# Load generated samples
print("\n=== Generated Samples Evaluation ===")
print("Loading generated samples...")
gen_data = np.load('generated_data/cvae_samples.npz')

# Combine all generated samples
X_gen_list = []
y_gen_list = []

for i, cat in enumerate(category_names):
    samples = gen_data[cat]
    X_gen_list.append(samples)
    y_gen_list.append(np.full(len(samples), i))

X_gen = np.vstack(X_gen_list)
y_gen = np.concatenate(y_gen_list)

print(f"Generated samples: {len(X_gen)}\n")

# Classify generated samples
y_pred_gen = clf.predict(X_gen)
acc_gen = accuracy_score(y_gen, y_pred_gen)

print(f"Accuracy on generated data: {acc_gen:.4f}")
if acc_gen >= 0.80:
    print("SUCCESS: Meets 80% threshold from proposal!")
else:
    print(f"Below 80% threshold (gap: {0.80 - acc_gen:.4f})")

print("\nPer-class accuracy on generated samples:")
for i, cat in enumerate(category_names):
    mask = y_gen == i
    acc = accuracy_score(y_gen[mask], y_pred_gen[mask])
    print(f"  {cat}: {acc:.4f}")

# Confusion matrix for generated samples
print("\nConfusion Matrix for Generated Samples:")
cm = confusion_matrix(y_gen, y_pred_gen)
print(cm)

# Visualize confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=category_names, yticklabels=category_names, ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix - Generated Samples')
plt.tight_layout()
plt.savefig('plots/cvae/confusion_matrix_generated.png', dpi=200)
print("\nSaved confusion matrix to plots/cvae/confusion_matrix_generated.png")

# Detailed classification report
print("\n=== Classification Report ===")
print(classification_report(y_gen, y_pred_gen, target_names=category_names))

# Summary
print("\n=== Summary ===")
print(f"Real data accuracy: {acc_real:.4f}")
print(f"Generated data accuracy: {acc_gen:.4f}")
print(f"Difference: {acc_real - acc_gen:.4f}")