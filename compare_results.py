"""
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Compare Original vs Balanced C-GAN

This script helps visualize the improvement from uniform label sampling
"""

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from cgan import ConditionalGAN

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load test data
data = torch.load('preprocessed_data/nslkdd_processed.pt', weights_only=False)
X_test = data['X_test'].numpy()
y_test = data['y_test'].numpy()
input_dim = data['input_dim']
num_classes = data['num_classes']
class_names = ['DoS', 'Normal', 'Probe', 'R2L', 'U2R']

# Train classifier on real data
print("Training classifier on real test data...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_test, y_test)
print()

def evaluate_model(model_path, model_name):
    """Generate samples and evaluate"""
    print(f"Evaluating {model_name}...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    noise_dim = checkpoint['noise_dim']
    
    model = ConditionalGAN(input_dim, num_classes, noise_dim).to(device)
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model.generator.eval()
    
    # Generate equal samples per class
    samples_per_class = 100
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
    
    # Evaluate
    y_pred = clf.predict(generated_data)
    cm = confusion_matrix(generated_labels, y_pred)
    
    # Per-class accuracy
    per_class_acc = []
    for i in range(num_classes):
        class_mask = (generated_labels == i)
        acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
        per_class_acc.append(acc)
    
    # Macro-average
    macro_avg = np.mean(per_class_acc)
    
    print(f"  Macro-Average: {macro_avg*100:.1f}%")
    for i, name in enumerate(class_names):
        print(f"    {name}: {per_class_acc[i]*100:.1f}%")
    print()
    
    return cm, per_class_acc, macro_avg

# Evaluate both models
print("="*60)
cm_original, acc_original, macro_original = evaluate_model(
    'models/cgan/cgan_noise100.pt', 
    'Original C-GAN (natural distribution)'
)

cm_balanced, acc_balanced, macro_balanced = evaluate_model(
    'models/cgan/cgan_balanced_noise100.pt',
    'Balanced C-GAN (uniform sampling)'
)
print("="*60)

# Create comparison visualization
fig = plt.figure(figsize=(16, 10))

# Confusion matrices
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax1)
ax1.set_title('Original C-GAN\nConfusion Matrix')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')

ax2 = plt.subplot(2, 3, 2)
sns.heatmap(cm_balanced, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names, ax=ax2)
ax2.set_title('Balanced C-GAN\nConfusion Matrix')
ax2.set_ylabel('True Label')
ax2.set_xlabel('Predicted Label')

# Per-class accuracy comparison
ax3 = plt.subplot(2, 3, 3)
x = np.arange(len(class_names))
width = 0.35
ax3.bar(x - width/2, [a*100 for a in acc_original], width, 
        label='Original', alpha=0.7, color='#3498db')
ax3.bar(x + width/2, [a*100 for a in acc_balanced], width,
        label='Balanced', alpha=0.7, color='#2ecc71')
ax3.axhline(y=80, color='gray', linestyle='--', label='Target (80%)')
ax3.set_ylabel('Accuracy (%)')
ax3.set_title('Per-Class Accuracy Comparison')
ax3.set_xticks(x)
ax3.set_xticklabels(class_names)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Macro-average comparison
ax4 = plt.subplot(2, 3, 4)
models = ['Original', 'Balanced']
macro_scores = [macro_original*100, macro_balanced*100]
colors = ['#e74c3c' if s < 80 else '#2ecc71' for s in macro_scores]
bars = ax4.bar(models, macro_scores, color=colors, alpha=0.7)
ax4.axhline(y=80, color='gray', linestyle='--', label='Target')
ax4.set_ylabel('Macro-Average Accuracy (%)')
ax4.set_title('Overall Performance\n(Macro-Average)')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Improvement breakdown
ax5 = plt.subplot(2, 3, 5)
improvements = [(acc_balanced[i] - acc_original[i])*100 for i in range(num_classes)]
colors_imp = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
ax5.barh(class_names, improvements, color=colors_imp, alpha=0.7)
ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax5.set_xlabel('Accuracy Improvement (%)')
ax5.set_title('Per-Class Improvement\n(Balanced - Original)')
ax5.grid(axis='x', alpha=0.3)

# Summary text
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
RESULTS SUMMARY

Original C-GAN:
  Macro-Avg: {macro_original*100:.1f}%
  Status: {'PASS' if macro_original >= 0.8 else 'FAIL'}
  
Balanced C-GAN:
  Macro-Avg: {macro_balanced*100:.1f}%
  Status: {'PASS' if macro_balanced >= 0.8 else 'FAIL'}

Improvement: {(macro_balanced - macro_original)*100:+.1f}%

Critical Classes:
  R2L: {acc_original[3]*100:.1f}% → {acc_balanced[3]*100:.1f}%
  U2R: {acc_original[4]*100:.1f}% → {acc_balanced[4]*100:.1f}%
"""
ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center')

plt.tight_layout()
plt.savefig('plots/cgan/comparison_original_vs_balanced.png', dpi=200, bbox_inches='tight')
print("Comparison plot saved to plots/cgan/comparison_original_vs_balanced.png")

# Print final summary
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(f"Original C-GAN Macro-Avg:  {macro_original*100:.1f}%")
print(f"Balanced C-GAN Macro-Avg:  {macro_balanced*100:.1f}%")
print(f"Improvement:               {(macro_balanced - macro_original)*100:+.1f}%")
print()
print("Per-Class Results:")
for i, name in enumerate(class_names):
    print(f"  {name:8s}: {acc_original[i]*100:5.1f}% → {acc_balanced[i]*100:5.1f}% ({(acc_balanced[i]-acc_original[i])*100:+.1f}%)")
print("="*60)