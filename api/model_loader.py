import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.ensemble import HybridPhysicsEnsembleAE

class ModelLoader:
    _instance = None
    _model = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, checkpoint_path: str = None):
        if self._model is not None:
            return self._model
        
        if checkpoint_path is None:
            # Default: look in project_root/checkpoints/best_model.pt
            checkpoint_path = project_root / "checkpoints" / "best_model.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        # Use config if available, else fallback to tiny mode defaults
        n_particles = config.get('n_particles', 50)
        latent_dim = config.get('latent_dim', 8)
        
        self._model = HybridPhysicsEnsembleAE(
            n_particles=n_particles,
            latent_dim=latent_dim
        )
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()
        
        self._config = config
        print(f"✅ Model loaded from {checkpoint_path}")
        return self._model
    
    def get_model(self):
        if self._model is None:
            self.load_model()
        return self._model
    
    def get_config(self):
        return self._config