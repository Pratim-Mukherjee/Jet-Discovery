import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp_ae import MLPAutoencoder
from .conv_ae import ConvAutoencoder
from .graph_ae import GraphAutoencoder
from physics.constraints import PhysicsConstraints

class HybridPhysicsEnsembleAE(nn.Module):
    """
    Hybrid Physics-Aware Ensemble Autoencoder.
    Combines MLP, Conv, and Graph autoencoders with physics constraints.
    """
    
    def __init__(
        self,
        n_particles: int = 200,
        latent_dim: int = 32,
        ensemble_weights: list = [0.4, 0.3, 0.3],
        physics_weights: dict = None
    ):
        super().__init__()
        
        self.n_particles = n_particles
        self.latent_dim = latent_dim
        self.ensemble_weights = ensemble_weights
        
        # Individual autoencoders
        self.mlp_ae = MLPAutoencoder(
            input_dim=n_particles * 4,
            latent_dim=latent_dim
        )
        self.conv_ae = ConvAutoencoder(
            n_particles=n_particles,
            latent_dim=latent_dim
        )
        self.graph_ae = GraphAutoencoder(
            n_particles=n_particles,
            latent_dim=latent_dim
        )
        
        # Physics constraints
        if physics_weights is None:
            physics_weights = {'energy_weight': 0.1, 'mass_weight': 0.05, 'pt_weight': 0.05}
        self.physics = PhysicsConstraints(**physics_weights)
        
        # Ensemble fusion layer (learned weighting)
        self.fusion_weights = nn.Parameter(torch.tensor(ensemble_weights))
    
    def forward(self, x, mask=None):
        """
        Forward pass through all three autoencoders.
        
        Args:
            x: [batch, n_particles, 4] input particles
            mask: [batch, n_particles] padding mask
        
        Returns:
            Dictionary with reconstructions, latents, and physics loss
        """
        # Get reconstructions from each model
        x_recon_mlp, z_mlp = self.mlp_ae(x)
        x_recon_conv, z_conv = self.conv_ae(x)
        x_recon_graph, z_graph = self.graph_ae(x)
        
        # Ensemble reconstruction (weighted average)
        weights = F.softmax(self.fusion_weights, dim=0)
        x_recon_ensemble = (
            weights[0] * x_recon_mlp +
            weights[1] * x_recon_conv +
            weights[2] * x_recon_graph
        )
        
        # Physics loss (returns per‑sample tensors)
        physics_losses = self.physics(x, x_recon_ensemble, mask)
        
        return {
            'recon_mlp': x_recon_mlp,
            'recon_conv': x_recon_conv,
            'recon_graph': x_recon_graph,
            'recon_ensemble': x_recon_ensemble,
            'z_mlp': z_mlp,
            'z_conv': z_conv,
            'z_graph': z_graph,
            'physics_loss': physics_losses,
            'ensemble_weights': weights
        }
    
    def get_anomaly_score(self, x, mask=None):
        """
        Compute anomaly score without batch normalisation.
        Returns raw scores that vary per sample.
        """
        outputs = self.forward(x, mask)

        # Reconstruction errors per model
        recon_mlp = F.mse_loss(outputs['recon_mlp'], x, reduction='none')
        recon_conv = F.mse_loss(outputs['recon_conv'], x, reduction='none')
        recon_graph = F.mse_loss(outputs['recon_graph'], x, reduction='none')

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            recon_mlp = recon_mlp * mask_expanded
            recon_conv = recon_conv * mask_expanded
            recon_graph = recon_graph * mask_expanded

        # Per‑sample mean reconstruction error (over all particles and features)
        recon_mlp = recon_mlp.mean(dim=(1, 2))
        recon_conv = recon_conv.mean(dim=(1, 2))
        recon_graph = recon_graph.mean(dim=(1, 2))

        weights = outputs['ensemble_weights']   # shape (3,)
        recon_ensemble = (
            weights[0] * recon_mlp +
            weights[1] * recon_conv +
            weights[2] * recon_graph
        )

        # Latent space score: L2 norm of latent vector (average over models)
        latent_score = (
            torch.norm(outputs['z_mlp'], dim=1) +
            torch.norm(outputs['z_conv'], dim=1) +
            torch.norm(outputs['z_graph'], dim=1)
        ) / 3

        # Physics score: total physics loss (per‑sample tensor)
        physics_score = outputs['physics_loss']['total']

        # Combine without batch normalisation
        total_score = (
            0.5 * recon_ensemble +
            0.3 * latent_score +
            0.2 * physics_score
        )

        return {
            'recon_error': recon_ensemble,
            'latent_score': latent_score,
            'physics_score': physics_score,
            'total_score': total_score,
            'recon_mlp': recon_mlp,
            'recon_conv': recon_conv,
            'recon_graph': recon_graph,
            'ensemble_weights': weights
        }
    
    def loss(self, x, mask=None, lambda_recon=1.0, lambda_physics=0.1):
        """
        Compute training loss (scalars for backpropagation).
        """
        outputs = self.forward(x, mask)
    
        # Reconstruction losses (per sample) then mean over batch
        recon_mlp = F.mse_loss(outputs['recon_mlp'], x, reduction='mean')
        recon_conv = F.mse_loss(outputs['recon_conv'], x, reduction='mean')
        recon_graph = F.mse_loss(outputs['recon_graph'], x, reduction='mean')
    
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            recon_mlp = F.mse_loss(outputs['recon_mlp'] * mask_expanded, x * mask_expanded, reduction='mean')
            recon_conv = F.mse_loss(outputs['recon_conv'] * mask_expanded, x * mask_expanded, reduction='mean')
            recon_graph = F.mse_loss(outputs['recon_graph'] * mask_expanded, x * mask_expanded, reduction='mean')
    
        weights = outputs['ensemble_weights']
        recon_ensemble = (
            weights[0] * recon_mlp +
            weights[1] * recon_conv +
            weights[2] * recon_graph
        )
    
        # Physics loss: now per‑sample, take mean over batch
        physics_loss = outputs['physics_loss']['total'].mean()
    
        total_loss = lambda_recon * recon_ensemble + lambda_physics * physics_loss
    
        return {
            'total': total_loss,
            'recon': recon_ensemble,
            'recon_mlp': recon_mlp,
            'recon_conv': recon_conv,
            'recon_graph': recon_graph,
            'physics': physics_loss,
            'weights': weights
        }