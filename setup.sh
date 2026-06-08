#!/bin/bash
# Setup script for Conditional Generative Models for Network Attack Synthesis
# Tested on SPHERE virtualgpu project with cuda126-ubuntu2404 image

echo "Setting up environment..."

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc

# Accept conda ToS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create environment
conda create -n genai python=3.10 -y
conda activate genai

# Install dependencies
# Note: torch==2.4.0 with cu121 wheels required for CUDA 12.8 driver on SPHERE
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn matplotlib seaborn pandas imbalanced-learn

echo "Environment setup complete!"
echo "To activate: source ~/miniconda/bin/activate genai"
