"""
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Macro-Average Evaluation

Key change: use macro-average metrics instead of overall accuracy
Macro-average treats all classes equally (important for imbalanced data)

Professor's suggestion: evaluate using metrics that don't favor majority classes
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from cgan import ConditionalGAN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load preprocessed test data
print("Loading test data...")
data = torch.load('preprocessed_data/nslkdd_processed.pt', weights_only=False)
X_test = data['X_test'].numpy()
y_test = data['y_test'].numpy()
input_dim = data['input_dim']
num_classes = data['num_classes']

print(f"Test samples: {len(X_test)}")
print(f"Number of classes: {num_classes}\n")

# Load trained model
print("Loading balanced C-GAN model...")
checkpoint = torch.load('models/cgan/cgan_noise100_epochs300.pt', 
                       map_location=device, weights_only=False)
noise_dim = checkpoint['noise_dim']

model = ConditionalGAN(input_dim, num_classes, noise_dim).to(device)
model.generator.load_state_dict(checkpoint['generator_state_dict'])
model.generator.eval()

print(f"Model loaded (trained for {checkpoint['epoch']} epochs)\n")

# Generate synthetic samples
# For evaluation, we want equal samples per class to match our training approach
samples_per_class = 100
print(f"Generating {samples_per_class} samples per class...")
print(f"Total synthetic samples: {samples_per_class * num_classes}\n")

generated_data = []
generated_labels = []

with torch.no_grad():
    for class_idx in range(num_classes):
        # Generate samples for this class
        labels = torch.full((samples_per_class,), class_idx, dtype=torch.long).to(device)
        z = torch.randn(samples_per_class, noise_dim).to(device)
        
        fake_samples = model.generator(z, labels)
        generated_data.append(fake_samples.cpu().numpy())
        generated_labels.extend([class_idx] * samples_per_class)

generated_data = np.vstack(generated_data)
generated_labels = np.array(generated_labels)

print(f"Generated data shape: {generated_data.shape}")
print(f"Generated labels shape: {generated_labels.shape}\n")

# Train classifier on real data
print("Training Random Forest classifier on real data...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_test, y_test)

# Evaluate on generated data
print("Evaluating synthetic samples...\n")
y_pred = clf.predict(generated_data)

# Confusion matrix
cm = confusion_matrix(generated_labels, y_pred)
print("Confusion Matrix:")
print(cm)
print()

# Per-class accuracy
print("Per-Class Accuracy:")
class_names = ['DoS', 'Normal', 'Probe', 'R2L', 'U2R']
per_class_acc = []
for i in range(num_classes):
    class_mask = (generated_labels == i)
    if class_mask.sum() > 0:
        acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
        per_class_acc.append(acc)
        print(f"  {class_names[i]}: {acc*100:.1f}%")
    else:
        per_class_acc.append(0.0)
        print(f"  {class_names[i]}: N/A (no samples)")

# KEY METRIC: Macro-average accuracy
# this is the average of per-class accuracies
# treats all classes equally regardless of sample count
macro_avg_acc = np.mean(per_class_acc)
print(f"\nMacro-Average Accuracy: {macro_avg_acc*100:.1f}%")
print("(average of per-class accuracies - treats all classes equally)")

# For comparison, also show regular accuracy
overall_acc = (y_pred == generated_labels).sum() / len(generated_labels)
print(f"Overall Accuracy: {overall_acc*100:.1f}%")
print("(total correct / total samples - can be misleading with imbalance)")
print()

# Classification report with macro averages
print("Classification Report:")
print(classification_report(generated_labels, y_pred, 
                          target_names=class_names,
                          zero_division=0))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix - Balanced C-GAN Generated Samples')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('plots/cgan/cgan_noise100_epochs300_confusion_matrix.png', dpi=200)
print("Confusion matrix saved to plots/cgan/cgan_noise100_epochs300_confusion_matrix.png")

# Plot per-class accuracy comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot of per-class accuracies
colors = ['#2ecc71' if acc >= 0.8 else '#e74c3c' for acc in per_class_acc]
ax1.bar(class_names, [acc*100 for acc in per_class_acc], color=colors, alpha=0.7)
ax1.axhline(y=80, color='gray', linestyle='--', label='80% Target')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Per-Class Accuracy (Balanced C-GAN)')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add macro-average line
ax1.axhline(y=macro_avg_acc*100, color='blue', linestyle='-', 
            linewidth=2, label=f'Macro-Avg: {macro_avg_acc*100:.1f}%')
ax1.legend()

# Distribution of generated samples vs true labels
class_counts_gen = [sum(generated_labels == i) for i in range(num_classes)]
class_counts_pred = [sum(y_pred == i) for i in range(num_classes)]

x = np.arange(len(class_names))
width = 0.35

ax2.bar(x - width/2, class_counts_gen, width, label='Generated', alpha=0.7)
ax2.bar(x + width/2, class_counts_pred, width, label='Predicted', alpha=0.7)
ax2.set_ylabel('Sample Count')
ax2.set_title('Generated vs Predicted Class Distribution')
ax2.set_xticks(x)
ax2.set_xticklabels(class_names)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/cgan/cgan_noise100_epochs300_class_accuracy.png', dpi=200)
print("Accuracy plots saved to plots/cgan/cgan_noise100_epochs300_class_accuracy.png")

# Summary for project report
print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)
print(f"Macro-Average Accuracy: {macro_avg_acc*100:.1f}%")
print(f"Target: 80%")
print(f"Status: {'PASS ✓' if macro_avg_acc >= 0.8 else 'FAIL ✗'}")
print()
print("Per-class breakdown:")
for i, name in enumerate(class_names):
    status = "✓" if per_class_acc[i] >= 0.8 else "✗"
    print(f"  {name}: {per_class_acc[i]*100:.1f}% {status}")
print("="*60)