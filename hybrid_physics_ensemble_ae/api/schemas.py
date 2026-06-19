from pydantic import BaseModel, Field
from typing import List, Optional

class ParticleInput(BaseModel):
    """Single particle input."""
    E: float = Field(..., description="Energy")
    px: float = Field(..., description="Momentum x")
    py: float = Field(..., description="Momentum y")
    pz: float = Field(..., description="Momentum z")

class JetInput(BaseModel):
    """Jet input with up to 200 particles."""
    particles: List[ParticleInput] = Field(..., max_items=200)
    
    def to_tensor(self, max_particles=200):
        """Convert to tensor format."""
        import torch
        n = len(self.particles)
        tensor = torch.zeros(max_particles, 4)
        for i, p in enumerate(self.particles[:max_particles]):
            tensor[i] = torch.tensor([p.E, p.px, p.py, p.pz])
        return tensor, n

class AnomalyResponse(BaseModel):
    """Anomaly detection response."""
    jet_id: Optional[str] = None
    total_anomaly_score: float = Field(..., description="Combined anomaly score (higher = more anomalous)")
    reconstruction_error: float = Field(..., description="Reconstruction error")
    latent_score: float = Field(..., description="Latent space deviation")
    physics_score: float = Field(..., description="Physics constraint violation")
    is_anomaly: bool = Field(..., description="Whether this jet is flagged as anomalous")
    threshold: float = Field(..., description="Anomaly threshold used")
    
    # Breakdown
    recon_mlp: Optional[float] = None
    recon_conv: Optional[float] = None
    recon_graph: Optional[float] = None
    ensemble_weights: Optional[List[float]] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    n_particles: int
    latent_dim: int