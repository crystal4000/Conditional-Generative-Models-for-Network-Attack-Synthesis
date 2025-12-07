"""
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Data Preprocessing

References:
- Conditional VAE slides 156-158 for conditioning approach
- Need to concatenate one-hot labels with features (conditioning point 1 in our architecture diagram)
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

# Same setup as explore script
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty'
]

attack_mapping = {
    'normal': 'Normal',
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS', 'processtable': 'DoS',
    'mailbomb': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}

print("\nPreprocessing NSL-KDD for Conditional Generative Models\n")

# Load data
print("Loading datasets...")
train_df = pd.read_csv('NSL-KDD-Dataset\KDDTrain+.txt', names=column_names, header=None)
test_df = pd.read_csv('NSL-KDD-Dataset\KDDTest+.txt', names=column_names, header=None)

# Map to categories
train_df['category'] = train_df['label'].map(attack_mapping)
test_df['category'] = test_df['label'].map(attack_mapping)

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Split features and labels
categorical_cols = ['protocol_type', 'service', 'flag']
numerical_cols = [col for col in train_df.columns 
                  if col not in categorical_cols + ['label', 'difficulty', 'category']]

print(f"\nFeature counts:")
print(f"  Categorical: {len(categorical_cols)}")
print(f"  Numerical: {len(numerical_cols)}")

# Step 1: One-hot encode categorical features
print("\n[1] One-hot encoding categorical features...")
train_cat = pd.get_dummies(train_df[categorical_cols], prefix=categorical_cols)
test_cat = pd.get_dummies(test_df[categorical_cols], prefix=categorical_cols)

# Make sure train and test have same columns after one-hot encoding
# test set might have different categories
train_cat, test_cat = train_cat.align(test_cat, join='left', axis=1, fill_value=0)

# Convert to numeric to avoid object dtype issues
train_cat = train_cat.astype(np.float32)
test_cat = test_cat.astype(np.float32)

print(f"  Categorical features after one-hot: {train_cat.shape[1]}")

# Step 2: Normalize numerical features
# using StandardScaler like we did in HW2 for VAE
print("\n[2] Normalizing numerical features...")
scaler = StandardScaler()
train_num = scaler.fit_transform(train_df[numerical_cols])
test_num = scaler.transform(test_df[numerical_cols])

print(f"  Numerical features: {train_num.shape[1]}")

# Step 3: Combine features
print("\n[3] Combining features...")
X_train = np.concatenate([train_num, train_cat.values], axis=1)
X_test = np.concatenate([test_num, test_cat.values], axis=1)

print(f"  Total feature dimension: {X_train.shape[1]}")

# Step 4: Encode category labels
# these will be our conditioning signals (y in the C-VAE/C-GAN)
print("\n[4] Encoding attack categories...")
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['category'])
y_test = label_encoder.transform(test_df['category'])

category_names = label_encoder.classes_
print(f"  Categories: {list(category_names)}")
print(f"  Encoded as: {list(range(len(category_names)))}")

# Check distribution in encoded form
print("\n  Training distribution:")
for i, cat in enumerate(category_names):
    count = (y_train == i).sum()
    print(f"    {cat} (label {i}): {count} samples")

# Step 5: Create validation split from training data
# we'll use 20% for validation like in the proposal
print("\n[5] Creating validation split...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"  Train: {X_train.shape[0]} samples")
print(f"  Val: {X_val.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")

# Step 6: Convert to PyTorch tensors
print("\n[6] Converting to PyTorch tensors...")
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)

X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Step 7: Save everything
os.makedirs('preprocessed_data', exist_ok=True)
print("\n[7] Saving preprocessed data...")

# Save tensors
torch.save({
    'X_train': X_train_tensor,
    'y_train': y_train_tensor,
    'X_val': X_val_tensor,
    'y_val': y_val_tensor,
    'X_test': X_test_tensor,
    'y_test': y_test_tensor,
    'input_dim': X_train.shape[1],
    'num_classes': len(category_names),
    'category_names': category_names
}, 'preprocessed_data/nslkdd_processed.pt')

# Save the scaler and encoder so we can inverse transform later if needed
with open('preprocessed_data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('preprocessed_data/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\nSaved files:")
print("  - preprocessed_data/nslkdd_processed.pt (main data)")
print("  - preprocessed_data/scaler.pkl (for inverse transform)")
print("  - preprocessed_data/label_encoder.pkl (for category names)")

print(f"\nDataset ready for training!")
print(f"Input dimension: {X_train.shape[1]}")
print(f"Number of classes: {len(category_names)}")
print(f"Classes: {list(category_names)}")