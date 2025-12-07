"""
bagan_gp.py
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - BAGAN-GP Architecture

BAGAN-GP combines:
1. Autoencoder initialization (gives generator common knowledge)
2. Gradient penalty (stabilizes training)
3. Balanced sampling (uniform labels during training)

Key insight from BAGAN-GP paper:
"Autoencoder learns general data structure from all classes,
then GAN refines generation quality through adversarial training"

References:
- BAGAN-GP (2021): autoencoder initialization + gradient penalty
- Gradient penalty from WGAN-GP helps with rare classes
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator initialized from pretrained decoder
    Takes noise + class label, outputs synthetic sample
    """
    def __init__(self, noise_dim, num_classes, latent_dim, output_dim):
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Project noise to latent dimension
        # this replaces the encoder in autoencoder
        self.noise_to_latent = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder part (will be initialized from pretrained autoencoder)
        # takes latent + class and produces synthetic sample
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, noise, labels):
        # Convert labels to one-hot
        labels_onehot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        
        # Concatenate noise with label
        noise_with_label = torch.cat([noise, labels_onehot], dim=1)
        
        # Project to latent space
        latent = self.noise_to_latent(noise_with_label)
        
        # Concatenate latent with label for decoder
        latent_with_label = torch.cat([latent, labels_onehot], dim=1)
        
        # Generate sample
        output = self.decoder(latent_with_label)
        
        return output

class Discriminator(nn.Module):
    """
    Discriminator initialized from pretrained encoder
    Takes sample + class label, outputs real/fake probability
    """
    def __init__(self, input_dim, num_classes, latent_dim):
        super(Discriminator, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Encoder part (will be initialized from pretrained autoencoder)
        # learns to extract features from samples
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Discriminator head
        # maps from latent representation to real/fake decision
        self.discriminator_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        # Convert labels to one-hot
        labels_onehot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        
        # Concatenate input with label
        x_with_label = torch.cat([x, labels_onehot], dim=1)
        
        # Encode to latent space
        latent = self.encoder(x_with_label)
        
        # Discriminator decision
        output = self.discriminator_head(latent)
        
        return output

class BAGAN_GP(nn.Module):
    """
    Complete BAGAN-GP model combining Generator and Discriminator
    """
    def __init__(self, input_dim, num_classes, noise_dim=100, latent_dim=128):
        super(BAGAN_GP, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        
        self.generator = Generator(noise_dim, num_classes, latent_dim, input_dim)
        self.discriminator = Discriminator(input_dim, num_classes, latent_dim)
    
    def load_pretrained_weights(self, autoencoder_path):
        """
        Load weights from pretrained autoencoder
        Encoder weights -> Discriminator encoder
        Decoder weights -> Generator decoder
        """
        print(f"Loading pretrained weights from {autoencoder_path}")
        checkpoint = torch.load(autoencoder_path, map_location='cpu', weights_only=False)
        
        # Load encoder weights into discriminator
        self.discriminator.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        print("  Loaded encoder -> discriminator")
        
        # Load decoder weights into generator
        self.generator.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print("  Loaded decoder -> generator")
        
        print("Autoencoder weights loaded successfully!")
        print("Generator and Discriminator now have common knowledge from pre-training\n")

def compute_gradient_penalty(discriminator, real_samples, fake_samples, labels, device):
    """
    Gradient penalty for WGAN-GP
    Helps stabilize training especially for rare classes
    
    From WGAN-GP paper: penalize gradients that deviate from norm 1
    GP = λ * E[(||∇D(x̂)||₂ - 1)²]
    
    This prevents discriminator from becoming too confident,
    which is critical when it has seen very few real U2R samples
    """
    batch_size = real_samples.size(0)
    
    # Random weight for interpolation
    alpha = torch.rand(batch_size, 1).to(device)
    
    # Interpolate between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # Get discriminator output on interpolated samples
    d_interpolates = discriminator(interpolates, labels)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Flatten gradients
    gradients = gradients.view(batch_size, -1)
    
    # Compute gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty