import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsConstraints(nn.Module):
    """
    Physics-aware constraints for jet reconstruction.
    Enforces energy-momentum conservation and mass consistency.
    """
    
    def __init__(self, energy_weight=0.1, mass_weight=0.05, pt_weight=0.05):
        super().__init__()
        self.energy_weight = energy_weight
        self.mass_weight = mass_weight
        self.pt_weight = pt_weight
    
    def compute_jet_features(self, particles, mask=None):
        """
        Compute jet-level features from constituent particles.
        
        Args:
            particles: [batch, n_particles, 4] (E, px, py, pz)
            mask: [batch, n_particles] binary mask for padding
        
        Returns:
            dict with E, px, py, pz, mass, pt
        """
        if mask is None:
            mask = torch.ones_like(particles[..., 0])
        
        # Sum over particles (with mask)
        E = torch.sum(particles[..., 0] * mask, dim=1)
        px = torch.sum(particles[..., 1] * mask, dim=1)
        py = torch.sum(particles[..., 2] * mask, dim=1)
        pz = torch.sum(particles[..., 3] * mask, dim=1)
        
        # Invariant mass
        mass2 = E**2 - px**2 - py**2 - pz**2
        mass = torch.sqrt(torch.clamp(mass2, min=1e-8))
        
        # Transverse momentum
        pt = torch.sqrt(px**2 + py**2)
        
        return {
            'E': E, 'px': px, 'py': py, 'pz': pz,
            'mass': mass, 'pt': pt
        }
    
    def energy_momentum_loss(self, original, reconstructed, mask=None):
        """
        Enforce conservation of energy and momentum.
        """
        orig_feats = self.compute_jet_features(original, mask)
        recon_feats = self.compute_jet_features(reconstructed, mask)
        
        # MSE between jet-level features
        loss_E = F.mse_loss(orig_feats['E'], recon_feats['E'])
        loss_px = F.mse_loss(orig_feats['px'], recon_feats['px'])
        loss_py = F.mse_loss(orig_feats['py'], recon_feats['py'])
        loss_pz = F.mse_loss(orig_feats['pz'], recon_feats['pz'])
        
        return loss_E + loss_px + loss_py + loss_pz
    
    def mass_loss(self, original, reconstructed, mask=None):
        """Penalize deviation in invariant mass."""
        orig_feats = self.compute_jet_features(original, mask)
        recon_feats = self.compute_jet_features(reconstructed, mask)
        
        return F.mse_loss(orig_feats['mass'], recon_feats['mass'])
    
    def pt_loss(self, original, reconstructed, mask=None):
        """Penalize deviation in transverse momentum."""
        orig_feats = self.compute_jet_features(original, mask)
        recon_feats = self.compute_jet_features(reconstructed, mask)
        
        return F.mse_loss(orig_feats['pt'], recon_feats['pt'])
    
    def forward(self, original, reconstructed, mask=None):
        """
        Compute total physics loss.
        """
        loss_em = self.energy_momentum_loss(original, reconstructed, mask)
        loss_mass = self.mass_loss(original, reconstructed, mask)
        loss_pt = self.pt_loss(original, reconstructed, mask)
        
        total = (
            self.energy_weight * loss_em +
            self.mass_weight * loss_mass +
            self.pt_weight * loss_pt
        )
        
        return {
            'total': total,
            'energy_momentum': loss_em,
            'mass': loss_mass,
            'pt': loss_pt
        }