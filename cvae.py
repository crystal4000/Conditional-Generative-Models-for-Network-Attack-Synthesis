"""
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Conditional VAE Implementation

References:
- Lecture slides 156-159 on Conditional VAE
- Slides 115-116 for reparameterization trick
- Our architecture diagram from presentation (conditioning at two points)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim=64, hidden_dims=[512, 256]):
        """
        Conditional VAE for network attack generation
        Concatenates one-hot labels with input and latent vectors
        """
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Encoder takes [input + label]
        # following the architecture from proposal
        encoder_input_dim = input_dim + num_classes
        self.encoder_fc1 = nn.Linear(encoder_input_dim, hidden_dims[0])
        self.encoder_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        
        # mu and logvar for reparameterization trick (slide 115)
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)
        
        # Decoder takes [latent + label]
        decoder_input_dim = latent_dim + num_classes
        self.decoder_fc1 = nn.Linear(decoder_input_dim, hidden_dims[1])
        self.decoder_fc2 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.decoder_fc3 = nn.Linear(hidden_dims[0], input_dim)
    
    def encode(self, x, labels):
        """
        Encoder: q(z|x,y)
        Concatenate input with one-hot label (conditioning point 1)
        """
        # One-hot encode the labels
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Concatenate x and y
        x_cond = torch.cat([x, y_onehot], dim=1)
        
        # Forward through encoder
        h = F.relu(self.encoder_fc1(x_cond))
        h = F.relu(self.encoder_fc2(h))
        
        # Get mu and logvar for the latent distribution
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        From slide 115
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    def decode(self, z, labels):
        """
        Decoder: p(x|z,y)
        Concatenate latent with one-hot label (conditioning point 2)
        """
        # One-hot encode the labels
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Concatenate z and y
        z_cond = torch.cat([z, y_onehot], dim=1)
        
        # Forward through decoder
        h = F.relu(self.decoder_fc1(z_cond))
        h = F.relu(self.decoder_fc2(h))
        x_recon = self.decoder_fc3(h)
        
        return x_recon
    
    def forward(self, x, labels):
        """
        Full forward pass: encode -> reparameterize -> decode
        """
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, labels)
        
        return x_recon, mu, logvar
    
    def sample(self, num_samples, labels, device):
        """
        Generate samples by sampling from prior p(z) = N(0,I)
        Then decode with given labels
        """
        # Sample from standard normal
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        # Decode with conditioning
        samples = self.decode(z, labels)
        
        return samples


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    ELBO loss for VAE
    Loss = Reconstruction Loss + beta * KL Divergence
    
    From slides 116 (ELBO formulation)
    beta-VAE mentioned in slides 141 for controlling KL strength
    """
    # Reconstruction loss (MSE for continuous features)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence between q(z|x) and p(z) = N(0,I)
    # KL = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
    # from slide 116
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss (negative ELBO)
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss