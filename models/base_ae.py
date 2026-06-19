import torch
import torch.nn as nn

class BaseAutoencoder(nn.Module):
    """Base class for all autoencoders in the ensemble."""
    
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.name = "BaseAE"
    
    def encode(self, x):
        raise NotImplementedError
    
    def decode(self, z):
        raise NotImplementedError
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def get_anomaly_score(self, x, mask=None):
        """
        Compute reconstruction-based anomaly score.
        Lower score = more normal.
        """
        x_recon, z = self.forward(x)
        
        # Reconstruction error (MSE per sample)
        recon_error = torch.mean((x - x_recon) ** 2, dim=(1, 2))
        
        # Latent space deviation (distance from prior)
        latent_score = torch.norm(z, dim=1)
        
        return recon_error, latent_score, x_recon