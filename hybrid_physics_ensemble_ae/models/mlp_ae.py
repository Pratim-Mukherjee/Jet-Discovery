import torch
import torch.nn as nn
from .base_ae import BaseAutoencoder

class MLPAutoencoder(BaseAutoencoder):
    """
    Multi-Layer Perceptron Autoencoder.
    Processes flattened particle features.
    """
    
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[256, 128, 64]):
        super().__init__(input_dim, latent_dim)
        self.name = "MLP_AE"
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        # Flatten: [batch, n_particles, 4] -> [batch, n_particles * 4]
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return self.encoder(x_flat)
    
    def decode(self, z):
        x_flat = self.decoder(z)
        # Reshape back to [batch, n_particles, 4]
        return x_flat.view(-1, self.input_dim // 4, 4)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        z = self.encoder(x_flat)
        x_recon_flat = self.decoder(z)
        x_recon = x_recon_flat.view(batch_size, -1, 4)
        return x_recon, z