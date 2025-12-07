"""
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - NSL-KDD Dataset Exploration
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
# NSL-KDD has 41 features plus label and difficulty score
# based on the dataset documentation from our proposal
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

# Map specific attack names to 5 main categories
# these are our conditioning labels for C-VAE and C-GAN
attack_mapping = {
    'normal': 'Normal',
    # DoS
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
    'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS', 'processtable': 'DoS',
    'mailbomb': 'DoS',
    # Probe
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}

# Create a class to write to both console and log file
class TeeOutput:
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

# Set up logging to file
os.makedirs('dataset_analysis', exist_ok=True)
log_filename = f"dataset_analysis/exploration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_file = open(log_filename, 'w', encoding='utf-8')

# Redirect stdout to both console and log file
original_stdout = sys.stdout
sys.stdout = TeeOutput(sys.stdout, log_file)

print("\nNSL-KDD Dataset Exploration for Final Project\n")

# Load the dataset files
print("Loading training data...")
train_df = pd.read_csv('NSL-KDD-Dataset\KDDTrain+.txt', names=column_names, header=None)
print(f"Loaded {len(train_df)} training samples")

print("\nLoading test data...")
test_df = pd.read_csv('NSL-KDD-Dataset\KDDTest+.txt', names=column_names, header=None)
print(f"Loaded {len(test_df)} test samples")

print(f"\nDataset shapes:")
print(f"  Train: {train_df.shape}")
print(f"  Test: {test_df.shape}")

# Check what features we have
categorical_features = ['protocol_type', 'service', 'flag']
print(f"\nFeature breakdown:")
print(f"  Categorical features: {len(categorical_features)}")
print(f"  Numerical features: {41 - len(categorical_features)}")

# Any missing data?
print(f"\nMissing values check:")
print(f"  Train: {train_df.isnull().sum().sum()}")
print(f"  Test: {test_df.isnull().sum().sum()}")

# Look at the raw attack labels first
print(f"\nRaw attack labels in training set:")
print(train_df['label'].value_counts())
print(f"Total unique attack types: {train_df['label'].nunique()}")

# Map to our 5 categories (Normal, DoS, Probe, R2L, U2R)
train_df['category'] = train_df['label'].map(attack_mapping)
test_df['category'] = test_df['label'].map(attack_mapping)

# Check if any labels didn't map
unmapped_train = train_df[train_df['category'].isnull()]['label'].unique()
unmapped_test = test_df[test_df['category'].isnull()]['label'].unique()
if len(unmapped_train) > 0:
    print(f"\nUnmapped train labels: {unmapped_train}")
if len(unmapped_test) > 0:
    print(f"Unmapped test labels: {unmapped_test}")

# The class imbalance we need to address
print(f"\n--- Attack Category Distribution (Train) ---")
category_counts_train = train_df['category'].value_counts()
print(category_counts_train)
print(f"\nAs percentages:")
print((train_df['category'].value_counts(normalize=True) * 100).round(2))

print(f"\n--- Attack Category Distribution (Test) ---")
category_counts_test = test_df['category'].value_counts()
print(category_counts_test)
print(f"\nAs percentages:")
print((test_df['category'].value_counts(normalize=True) * 100).round(2))

# Visualization to show the imbalance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

category_counts_train.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Training Set Class Distribution')
ax1.set_xlabel('Attack Category')
ax1.set_ylabel('Number of Samples')
ax1.tick_params(axis='x', rotation=45)

category_counts_test.plot(kind='bar', ax=ax2, color='coral')
ax2.set_title('Test Set Class Distribution')
ax2.set_xlabel('Attack Category')
ax2.set_ylabel('Number of Samples')
ax2.tick_params(axis='x', rotation=45)

os.makedirs('dataset_analysis', exist_ok=True)
plt.tight_layout()
plt.savefig('dataset_analysis/class_distribution.png', dpi=200)
print("\nSaved visualization: dataset_analysis/class_distribution.png")

numerical_features = [col for col in train_df.columns 
                     if col not in categorical_features + ['label', 'difficulty', 'category']]
print(f"\nSample statistics for first 10 numerical features:")
print(train_df[numerical_features[:10]].describe())

print(f"\nCategorical feature values:")
for feat in categorical_features:
    print(f"\n{feat} - {train_df[feat].nunique()} unique values:")
    print(train_df[feat].value_counts().head())

print("\nDone with exploration!")

# Restore stdout and close log file
sys.stdout = original_stdout
log_file.close()
print(f"\nOutput saved to: {log_filename}")