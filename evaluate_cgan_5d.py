"""
evaluate_cgan_5d.py
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Evaluate 5:1 D:G Model
"""

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from cgan import ConditionalGAN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load test data
data = torch.load('preprocessed_data/nslkdd_processed.pt', weights_only=False)
X_test = data['X_test'].numpy()
y_test = data['y_test'].numpy()
input_dim = data['input_dim']
num_classes = data['num_classes']

print(f"Test samples: {len(X_test)}\n")

# Load model
print("Loading 5:1 D:G model...")
checkpoint = torch.load('models/cgan/cgan_balanced_5d_noise100.pt', 
                       map_location=device, weights_only=False)
noise_dim = checkpoint['noise_dim']

model = ConditionalGAN(input_dim, num_classes, noise_dim).to(device)
model.generator.load_state_dict(checkpoint['generator_state_dict'])
model.generator.eval()

print(f"Model trained for {checkpoint['epoch']} epochs\n")

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

# Train classifier
print("Training classifier on real data...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_test, y_test)

# Evaluate
print("Evaluating...\n")
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

# Classification report
print("\nClassification Report:")
print(classification_report(generated_labels, y_pred, 
                          target_names=class_names, zero_division=0))

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - 5:1 D:G Balanced C-GAN')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('plots/cgan/confusion_matrix_5d.png', dpi=200)
print("\nConfusion matrix saved to plots/cgan/confusion_matrix_5d.png")

# Summary
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