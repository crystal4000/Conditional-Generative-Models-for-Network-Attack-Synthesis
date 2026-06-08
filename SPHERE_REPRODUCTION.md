# Reproducing on SPHERE

## Requirements
- Access to SPHERE testbed (https://launch.sphere-testbed.net)
- Membership in the `virtualgpu` project

## Step 1 — Create the Experiment
1. Go to Experiments → Manage → New Experiment
2. Fill in:
   - Project: `virtualgpu`
   - Name: `attacksynth`
   - Description: "Reproducing conditional generative models for network attack synthesis on NSL-KDD"

## Step 2 — Push the Network Model
1. Go to Model Editor
2. Paste the following:

```python
from mergexp import *

net = Network('cgan-nslkdd', addressing==ipv4)
trainer = net.node('trainer', metal==True, image=="cuda126-ubuntu2404")
experiment(net)
```

3. Compile and push to `attacksynth.virtualgpu`, branch `master`

## Step 3 — Realize and Activate
1. Experiments → Manage → click `attacksynth`
2. Click the revision hash → Realize Revision → name it `run1`
3. Realizations → `run1` → ⋮ → Activate
4. Wait for Success (bare metal imaging takes 30-60 minutes)

## Step 4 — Set Up XDC
1. XDCs → New XDC
   - Project: `virtualgpu`
   - Name: any name
   - Type: personal
2. Wait for SSH Name to populate
3. ⋮ → Attach → select `run1.attacksynth.virtualgpu`

## Step 5 — Connect and Set Up Environment
1. Click the Jupyter link on your XDC
2. Start server on localhost
3. Open a Terminal
4. SSH into the trainer node:
```bash
ssh trainer
```
5. Clone the repo and run setup:
```bash
git clone https://github.com/crystal4000/Conditional-Generative-Models-for-Network-Attack-Synthesis.git
cd Conditional-Generative-Models-for-Network-Attack-Synthesis
bash setup.sh
source ~/miniconda/bin/activate genai
```

## Step 6 — Run the Pipeline
```bash
# Preprocess data
python preprocess_nslkdd.py

# Train C-GAN (300 epochs)
python train_cgan.py

# Train C-VAE
python train_cvae.py --beta 1.0 --latent_dim 64 --epochs 50
python train_cvae.py --beta 4.0 --latent_dim 64 --epochs 50

# Pretrain autoencoder for BAGAN-GP
python pretrain_autoencoder.py

# Train BAGAN-GP
python train_bagan_gp.py
```

## Notes
- Use tmux to keep training sessions alive: `tmux new -s training`
- Detach with Ctrl+B then D, reattach with `tmux attach -s training`
- Training runs on Quadro RTX 6000 (24GB VRAM)
- Full pipeline takes approximately 2-3 hours on GPU

## SPHERE Experiment Details
- Experiment: `attacksynth.virtualgpu`
- Model revision: `6ab28e9a063bb54774e47bc602d5c691f870b194`
- Node image: `cuda126-ubuntu2404`
- Facility: `gpuml`
- Resource: `sgpa1` (Quadro RTX 6000)
