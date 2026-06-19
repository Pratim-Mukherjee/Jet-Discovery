import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.ensemble import HybridPhysicsEnsembleAE

class ModelLoader:
    """Singleton model loader for FastAPI."""
    
    _instance = None
    _model = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, checkpoint_path: str = 'checkpoints/best_model.pt'):
        """Load the trained model."""
        if self._model is not None:
            return self._model
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        self._model = HybridPhysicsEnsembleAE(
            n_particles=config.get('n_particles', 200),
            latent_dim=config.get('latent_dim', 32)
        )
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()
        
        self._config = config
        return self._model
    
    def get_model(self):
        if self._model is None:
            self.load_model()
        return self._model
    
    def get_config(self):
        return self._config