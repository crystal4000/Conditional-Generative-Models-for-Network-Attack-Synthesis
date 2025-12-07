# Conditional Generative Models for Network Attack Synthesis

**CS 5331 - Generative AI | Final Project**  
**Team 11**: Tania Amanda Nkoyo Fredrick Eneye, Richard Linn

---

## Table of Contents
- [Overview](#overview)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project investigates whether conditional generative models can synthesize distinguishable network attack samples under extreme class imbalance. We implemented Conditional Variational Autoencoders (C-VAE) and Conditional Generative Adversarial Networks (C-GAN) on the NSL-KDD intrusion detection dataset.

### Problem Statement
Network intrusion detection systems require diverse training data, but real attack datasets exhibit extreme class imbalance:
- **Normal traffic**: 67,343 samples (53.46%)
- **DoS attacks**: 45,927 samples (36.46%)
- **Probe attacks**: 11,656 samples (9.25%)
- **R2L attacks**: 995 samples (0.79%)
- **U2R attacks**: 52 samples (0.04%)

This 1,300:1 imbalance ratio makes it nearly impossible to train classifiers that reliably detect rare attack types.

### Research Question
Can conditional generative models produce distinguishable synthetic samples for rare attack classes when given fewer than 100 training examples?

---

## Key Findings

### Main Discovery
**Conditional generative models require approximately 50-100 training samples per class to generate distinguishable synthetic data.**

### Experimental Results
| Approach | Macro-Avg Accuracy | DoS | Normal | Probe | R2L | U2R |
|----------|-------------------|-----|--------|-------|-----|-----|
| C-VAE (β=1.0) | 49.8% | 95% | 100% | 51% | 3% | 0% |
| C-VAE (β=4.0) | 46.2% | 83% | 100% | 48% | 0% | 0% |
| **C-GAN Baseline** | **59.6%** | **100%** | **100%** | **98%** | **0%** | **0%** |
| Uniform Sampling | 42.6% | 100% | 100% | 0% | 13% | 0% |
| BAGAN-GP | 49.8% | 100% | 100% | 49% | 0% | 0% |
| SMOTE-GAN | 37.2% | 100% | 77% | 9% | 0% | 0% |
| Extended (300 epochs) | 59.4% | 100% | 100% | 97% | 0% | 0% |
| 5:1 D:G Ratio | 40.4% | 98% | 98% | 6% | 0% | 0% |

**Target**: 80% macro-average accuracy  
**Outcome**: No approach succeeded on rare classes (R2L, U2R)

### What Worked
- Perfect generation of majority classes (DoS, Normal, Probe) when given 9,000+ training samples
- Stable training dynamics with proper architecture design
- Successful conditional control when sufficient data exists

### What Failed
- All approaches achieved 0% accuracy on U2R class (52 training samples)
- R2L class achieved at most 13% accuracy (995 training samples)
- Advanced techniques (BAGAN-GP, SMOTE) did not overcome fundamental data scarcity

---

## Dataset

### NSL-KDD
We use the NSL-KDD dataset, an improved version of the KDD Cup 99 dataset.

**Statistics**:
- Training: 125,973 samples
- Test: 22,544 samples
- Features: 41 (expanded to 122 after one-hot encoding)
- Classes: 5 (Normal, DoS, Probe, R2L, U2R)

**Download**: [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)

Place the following files in `data/`:
- `KDDTrain+.txt`
- `KDDTest+.txt`

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cgan-network-attacks.git
cd cgan-network-attacks
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate genai
```

Or install via pip:
```bash
pip install -r requirements.txt
```

3. Download NSL-KDD dataset and place in `data/` directory

---

## Project Structure
```
.
├── data/
│   ├── KDDTrain+.txt              # NSL-KDD training data
│   └── KDDTest+.txt               # NSL-KDD test data
├── preprocessed_data/
│   ├── nslkdd_processed.pt        # Preprocessed dataset
│   └── nslkdd_smote_augmented.pt  # SMOTE-augmented dataset
├── models/
│   ├── cgan/                      # C-GAN model checkpoints
│   ├── cvae/                      # C-VAE model checkpoints
│   ├── bagan/                     # BAGAN-GP model checkpoints
│   └── smote_gan/                 # SMOTE-GAN model checkpoints
├── plots/
│   ├── cgan/                      # C-GAN visualizations
│   ├── cvae/                      # C-VAE visualizations
│   ├── bagan/                     # BAGAN-GP visualizations
│   └── smote/                     # SMOTE visualizations
├── src/
│   ├── preprocess_nslkdd.py       # Data preprocessing
│   ├── cvae.py                    # C-VAE architecture
│   ├── cgan.py                    # C-GAN architecture
│   ├── bagan_gp.py                # BAGAN-GP architecture
│   ├── train_cvae.py              # C-VAE training
│   ├── train_cgan.py              # C-GAN training
│   ├── train_bagan_gp.py          # BAGAN-GP training
│   ├── smote_augmentation.py      # SMOTE preprocessing
│   ├── evaluate_cvae.py           # C-VAE evaluation
│   └── evaluate_cgan.py           # C-GAN evaluation
├── project_proposal/
│   └── Conditional Generative Models for Network Attack Synthesist.pptx           # Project presentation slides
├── report/
│   └── Conditional Generative Models for Network Attack Synthesis.pdf             # Project report
│   └── Conditional Generative Models for Network Attack Synthesis.zip             # Latex Source Code
├── environment.yml                # Conda environment
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Usage

### 1. Data Preprocessing
```bash
python preprocess_nslkdd.py
```

This script:
- One-hot encodes categorical features (protocol, service, flag)
- Standardizes numerical features
- Creates train/validation/test splits
- Saves processed data to `preprocessed_data/nslkdd_processed.pt`

### 2. Training Models

#### C-VAE (β=1.0)
```bash
python train_cvae.py --beta 1.0 --latent_dim 64 --epochs 50
```

#### C-VAE (β=4.0)
```bash
python train_cvae.py --beta 4.0 --latent_dim 64 --epochs 50
```

#### C-GAN Baseline
```bash
python train_cgan.py --noise_dim 100 --epochs 100
```

#### BAGAN-GP
```bash
# Phase 1: Pre-train autoencoder
python pretrain_autoencoder.py --epochs 100

# Phase 2: Train GAN with initialization
python train_bagan_gp.py --epochs 150
```

#### SMOTE-GAN
```bash
# Step 1: Apply SMOTE
python smote_augmentation.py

# Step 2: Train GAN on augmented data
python train_smote_gan.py --epochs 100
```

### 3. Evaluation
```bash
# Evaluate C-VAE
python evaluate_cvae.py --model_path models/cvae/cvae_beta1.0.pt

# Evaluate C-GAN
python evaluate_cgan.py --model_path models/cgan/cgan_noise100.pt

# Evaluate BAGAN-GP
python evaluate_bagan_gp.py --model_path models/bagan/bagan_gp_final.pt
```

### 4. Generate Samples
```bash
# Generate 100 samples per class
python sample_cgan.py --model_path models/cgan/cgan_noise100.pt \
                      --samples_per_class 100 \
                      --output_dir generated_data/
```

---

## Experiments

We systematically tested six approaches:

### 1. Baseline Models
- **C-VAE** with two β configurations (1.0, 4.0)
- **C-GAN** with standard training (100 epochs)

### 2. Uniform Label Sampling
Sample labels uniformly (20% per class) instead of from data distribution during training.

**Hypothesis**: Generator needs equal exposure to all classes.  
**Result**: Failed. Caused Probe collapse while slightly improving R2L (0% → 13%).

### 3. BAGAN-GP (Autoencoder Initialization)
Pre-train supervised autoencoder on all data, then initialize GAN from these weights.

**Hypothesis**: Transfer learning from majority classes provides "common knowledge".  
**Result**: Failed. Training was stable but rare class generation remained 0%.

### 4. SMOTE Preprocessing
Expand R2L (796 → 5,000) and U2R (52 → 500) using SMOTE before GAN training.

**Hypothesis**: More training samples (even synthetic) helps GAN learn.  
**Result**: Failed catastrophically. Discriminator detected SMOTE interpolations as unrealistic.

### 5. Extended Training
Train baseline C-GAN for 300 epochs instead of 100.

**Hypothesis**: Longer training allows generator to learn rare patterns.  
**Result**: Failed. Model converged by epoch 120 with no further improvement.

### 6. Modified Discriminator Ratio
Update discriminator 5 times per generator update instead of 1:1.

**Hypothesis**: Stronger discriminator provides better gradients for rare classes.  
**Result**: Failed. Caused discriminator saturation (loss spike to 20).

---

## Results

### Quantitative Results

**Best Model**: C-GAN Baseline (59.6% macro-average)

**Per-Class Breakdown**:
- ✅ DoS: 100% (45,927 training samples)
- ✅ Normal: 100% (67,343 training samples)
- ✅ Probe: 98% (11,656 training samples)
- ❌ R2L: 0% (995 training samples)
- ❌ U2R: 0% (52 training samples)

### Key Observations

1. **Clear Threshold**: Classes with 9,000+ samples achieved 97-100% accuracy. Classes with <1,000 samples achieved 0-13%.

2. **No Technique Helped**: Advanced approaches (BAGAN-GP, SMOTE) performed equal to or worse than baseline.

3. **Training Stability ≠ Generation Quality**: BAGAN-GP had the most stable training but didn't improve results.

4. **SMOTE Backfired**: Adding synthetic interpolations degraded performance across all classes.

### Empirical Finding

**Conditional generative models require approximately 50-100 training samples per class to generate distinguishable synthetic data.**

This threshold is consistent across:
- Different architectures (C-VAE, C-GAN, BAGAN-GP)
- Different training strategies (uniform sampling, extended training)
- Different data augmentation (SMOTE, autoencoder pre-training)

---

## Visualizations

All training curves, confusion matrices, and comparative plots are available in the `plots/` directory:

- Training loss curves for all models
- Confusion matrices showing collapse patterns
- Per-class accuracy comparisons
- Distribution visualizations (original vs SMOTE-augmented)
- Latent space t-SNE plots
- Sample quality comparisons

---

<!-- ## Citation

If you use this code or findings in your research, please cite:
```bibtex
@techreport{eneye2024cgan,
  title={Conditional Generative Models for Network Attack Synthesis: Investigating Limits Under Extreme Class Imbalance},
  author={Eneye, Tania Amanda Nkoyo Fredrick and Linn, Richard},
  institution={Texas Tech University},
  year={2024},
  type={CS 5331 Final Project Report}
}
``` -->

---

## Acknowledgments

- **Course**: CS 5331 - Generative AI, Texas Tech University
- **Instructor**: Dr. Stas Tiomkin
- **Dataset**: NSL-KDD from Canadian Institute for Cybersecurity
- **Frameworks**: PyTorch, scikit-learn, imbalanced-learn

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaboration:
- Tania-Amanda Fredrick Eneye: tafredri@ttu.edu
- Richard Linn: richard.linn@ttu.edu

---

## Future Work
Based on our findings, future research directions include:
1. **Data Collection**: Invest in collecting additional real attack samples via honeypots
2. **Alternative Approaches**: Explore cost-sensitive learning, one-class classification, few-shot learning
3. **Diffusion Models**: Test whether diffusion models handle extreme imbalance better than GANs/VAEs
4. **Hierarchical Generation**: Generate broader attack categories instead of specific types
5. **Hybrid Pipelines**: Staged approach (SMOTE → refinement) rather than simultaneous

---

**Last Updated**: December 2024