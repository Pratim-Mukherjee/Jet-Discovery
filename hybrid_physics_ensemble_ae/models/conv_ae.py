import torch
import torch.nn as nn
from .base_ae import BaseAutoencoder

class ConvAutoencoder(BaseAutoencoder):
    """
    1D Convolutional Autoencoder for particle clouds.
    Treats particles as a sequence with 4 channels (E, px, py, pz).
    """
    
    def __init__(self, n_particles, latent_dim=32):
        super().__init__(n_particles * 4, latent_dim)
        self.n_particles = n_particles
        self.name = "Conv_AE"
        
        # Encoder: 1D Conv layers
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Latent projection
        self.fc_mu = nn.Linear(128, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 128 * (n_particles // 4))
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, 4, kernel_size=3, padding=1),
            nn.Tanh(),  # Output in [-1, 1] range
        )
        
        # Interpolate to exact size if needed
        self._target_len = n_particles
    
    def encode(self, x):
        # x: [batch, n_particles, 4] -> [batch, 4, n_particles]
        x = x.permute(0, 2, 1)
        
        h = self.encoder(x)  # [batch, 128, 1]
        h = h.squeeze(-1)     # [batch, 128]
        
        return self.fc_mu(h)
    
    def decode(self, z):
        h = self.fc_decode(z)  # [batch, 128 * (n_particles//4)]
        h = h.view(-1, 128, self.n_particles // 4)
        
        x_recon = self.decoder(h)  # [batch, 4, n_particles']
        
        # Interpolate to exact length
        if x_recon.shape[-1] != self.n_particles:
            x_recon = nn.functional.interpolate(
                x_recon, size=self.n_particles, mode='linear', align_corners=False
            )
        
        # [batch, n_particles, 4]
        return x_recon.permute(0, 2, 1)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z