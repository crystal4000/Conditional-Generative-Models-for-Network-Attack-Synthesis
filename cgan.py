"""
Team 11: [Tania Amanda Nkoyo Fredrick Eneye, Richard Linn]
CS 5331 Final Project - Conditional GAN Implementation

References:
- Lecture slides 167-172 on GANs (forger vs expert concept)
- Proposal architecture: Generator and Discriminator with conditioning
- Both networks get the label as input (concatenated)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, output_dim, hidden_dims=[256, 512]):
        """
        Generator: Takes noise + label, outputs fake sample
        G(z, y) -> x_fake
        """
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        
        # Input is [noise + one-hot label]
        input_dim = noise_dim + num_classes
        
        # Following proposal architecture
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        
    def forward(self, z, labels):
        """
        Generate fake samples conditioned on labels
        """
        # One-hot encode labels
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Concatenate noise and label
        x = torch.cat([z, y_onehot], dim=1)
        
        # Forward pass
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)  # no activation on output
        
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256]):
        """
        Discriminator: Takes sample + label, outputs real/fake probability
        D(x, y) -> [0, 1]
        """
        super(Discriminator, self).__init__()
        
        self.num_classes = num_classes
        
        # Input is [data + one-hot label]
        input_size = input_dim + num_classes
        
        # Following proposal architecture
        self.fc1 = nn.Linear(input_size, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        
    def forward(self, x, labels):
        """
        Classify sample as real or fake, conditioned on labels
        """
        # One-hot encode labels
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Concatenate input and label
        inp = torch.cat([x, y_onehot], dim=1)
        
        # Forward pass
        h = F.leaky_relu(self.fc1(inp), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        out = torch.sigmoid(self.fc3(h))
        
        return out


class ConditionalGAN(nn.Module):
    def __init__(self, input_dim, num_classes, noise_dim=100):
        """
        Wrapper for Generator and Discriminator
        """
        super(ConditionalGAN, self).__init__()
        
        self.generator = Generator(noise_dim, num_classes, input_dim)
        self.discriminator = Discriminator(input_dim, num_classes)
        self.noise_dim = noise_dim
        
    def generate(self, num_samples, labels, device):
        """
        Generate samples given labels
        """
        z = torch.randn(num_samples, self.noise_dim).to(device)
        fake_samples = self.generator(z, labels)
        return fake_samples