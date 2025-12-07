"""
evaluate_bagan_gp.py
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Evaluate BAGAN-GP

Test if autoencoder initialization + gradient penalty
improved rare class generation (R2L, U2R)

Expected improvement based on BAGAN-GP paper:
- R2L: 0-13% → 20-30%
- U2R: 0% → 10-20%
"""

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from bagan_gp import BAGAN_GP

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load test data
print("Loading test data...")
data = torch.load('preprocessed_data/nslkdd_processed.pt', weights_only=False)
X_test = data['X_test'].numpy()
y_test = data['y_test'].numpy()
input_dim = data['input_dim']
num_classes = data['num_classes']

print(f"Test samples: {len(X_test)}\n")

# Load model
print("Loading BAGAN-GP model...")
checkpoint = torch.load('models/bagan/bagan_gp_final.pt', 
                       map_location=device, weights_only=False)

noise_dim = checkpoint['noise_dim']
latent_dim = checkpoint['latent_dim']

model = BAGAN_GP(input_dim, num_classes, noise_dim, latent_dim).to(device)
model.generator.load_state_dict(checkpoint['generator_state_dict'])
model.generator.eval()

print(f"Model trained for {checkpoint['epoch']} epochs")
print(f"Final losses: G={checkpoint['g_loss']:.4f}, D={checkpoint['d_loss']:.4f}\n")

# Generate samples
samples_per_class = 100
class_names = ['DoS', 'Normal', 'Probe', 'R2L', 'U2R']

print(f"Generating {samples_per_class} samples per class...\n")

generated_data = []
generated_labels = []

with torch.no_grad():
    for class_idx in range(num_classes):
        labels = torch.full((samples_per_class,), class_idx, dtype=torch.long).to(device)
        z = torch.randn(samples_per_class, noise_dim).to(device)
        fake_samples = model.generator(z, labels)
        generated_data.append(fake_samples.cpu().numpy())
        generated_labels.extend([class_idx] * samples_per_class)

generated_data = np.vstack(generated_data)
generated_labels = np.array(generated_labels)

# Train classifier on real data
print("Training classifier on real data...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_test, y_test)

# Evaluate
print("Evaluating BAGAN-GP generated samples...\n")
y_pred = clf.predict(generated_data)
cm = confusion_matrix(generated_labels, y_pred)

print("Confusion Matrix:")
print(cm)
print()

# Per-class accuracy
print("Per-Class Accuracy:")
per_class_acc = []
for i in range(num_classes):
    class_mask = (generated_labels == i)
    acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
    per_class_acc.append(acc)
    print(f"  {class_names[i]}: {acc*100:.1f}%")

macro_avg_acc = np.mean(per_class_acc)
print(f"\nMacro-Average Accuracy: {macro_avg_acc*100:.1f}%")
print(f"Target: 80%")

# Classification report
print("\nClassification Report:")
print(classification_report(generated_labels, y_pred, 
                          target_names=class_names, zero_division=0))

# Plot confusion matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax1)
ax1.set_title('BAGAN-GP Confusion Matrix')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')

# Per-class accuracy bar plot
colors = ['#2ecc71' if acc >= 0.8 else '#e74c3c' for acc in per_class_acc]
ax2.bar(class_names, [acc*100 for acc in per_class_acc], color=colors, alpha=0.7)
ax2.axhline(y=80, color='gray', linestyle='--', label='Target (80%)')
ax2.axhline(y=macro_avg_acc*100, color='blue', linestyle='-', 
            linewidth=2, label=f'Macro-Avg: {macro_avg_acc*100:.1f}%')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Per-Class Accuracy')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/bagan/bagan_gp_evaluation.png', dpi=200)
print("\nPlots saved to plots/bagan/bagan_gp_evaluation.png")

# Summary
print("\n" + "="*60)
print("BAGAN-GP EVALUATION SUMMARY")
print("="*60)
print(f"Macro-Average Accuracy: {macro_avg_acc*100:.1f}%")
print(f"Target: 80%")
print(f"Status: {'PASS ✓' if macro_avg_acc >= 0.8 else 'FAIL ✗'}")
print()
print("Per-class breakdown:")
for i, name in enumerate(class_names):
    status = "✓" if per_class_acc[i] >= 0.8 else "✗"
    print(f"  {name}: {per_class_acc[i]*100:.1f}% {status}")
print()
print("Critical rare classes:")
print(f"  R2L (796 train samples): {per_class_acc[3]*100:.1f}%")
print(f"  U2R (42 train samples):  {per_class_acc[4]*100:.1f}%")
print("="*60)

# Save results for comparison
results = {
    'macro_avg': macro_avg_acc,
    'per_class_acc': per_class_acc,
    'class_names': class_names,
    'confusion_matrix': cm
}
np.save('plots/bagan/bagan_gp_results.npy', results)
print("\nResults saved for comparison")