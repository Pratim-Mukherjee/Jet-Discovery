from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from typing import List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from api.schemas import JetInput, AnomalyResponse, HealthResponse
from api.model_loader import ModelLoader

# Create FastAPI app
app = FastAPI(
    title="Hybrid Physics-Aware Ensemble Autoencoder API",
    description="""
    Anomaly detection for jet physics using a hybrid ensemble of autoencoders.
    
    This API detects unknown/new physics signatures (anomalies) in jet data
    by combining:
    - MLP Autoencoder
    - Convolutional Autoencoder  
    - Graph Autoencoder
    - Physics-aware constraints (energy-momentum conservation, mass, pT)
    """,
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model
model_loader = ModelLoader()
ANOMALY_THRESHOLD = 1.5  # Can be tuned based on validation data

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        model_loader.load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("API will run in demo mode.")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        model = model_loader.get_model()
        config = model_loader.get_config()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            device=str(next(model.parameters()).device),
            n_particles=model.n_particles,
            latent_dim=model.latent_dim
        )
    except Exception:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            device="unknown",
            n_particles=200,
            latent_dim=32
        )

@app.post("/predict", response_model=AnomalyResponse)
async def predict_anomaly(jet: JetInput):
    """
    Detect anomalies in a jet.
    
    Higher anomaly score indicates the jet is more likely to contain
    new physics / unknown signatures.
    """
    try:
        model = model_loader.get_model()
        device = next(model.parameters()).device
        
        # Convert input to tensor
        particles, n = jet.to_tensor(model.n_particles)
        particles = particles.unsqueeze(0).to(device)  # [1, n_particles, 4]
        
        # Mask for actual particles
        mask = torch.zeros(model.n_particles)
        mask[:n] = 1.0
        mask = mask.unsqueeze(0).to(device)
        
        # Get anomaly scores
        with torch.no_grad():
            scores = model.get_anomaly_score(particles, mask)
        
        # Determine if anomaly
        is_anomaly = scores['total_score'].item() > ANOMALY_THRESHOLD
        
        # Get ensemble weights
        outputs = model.forward(particles, mask)
        weights = outputs['ensemble_weights'].cpu().numpy().tolist()
        
        return AnomalyResponse(
            jet_id=None,
            total_anomaly_score=scores['total_score'].item(),
            reconstruction_error=scores['recon_error'].item(),
            latent_score=scores['latent_score'].item(),
            physics_score=scores['physics_score'].item(),
            is_anomaly=is_anomaly,
            threshold=ANOMALY_THRESHOLD,
            recon_mlp=scores['recon_mlp'].item(),
            recon_conv=scores['recon_conv'].item(),
            recon_graph=scores['recon_graph'].item(),
            ensemble_weights=weights
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(jets: List[JetInput]):
    """Batch anomaly detection."""
    try:
        model = model_loader.get_model()
        device = next(model.parameters()).device
        
        results = []
        for jet in jets:
            particles, n = jet.to_tensor(model.n_particles)
            particles = particles.unsqueeze(0).to(device)
            
            mask = torch.zeros(model.n_particles)
            mask[:n] = 1.0
            mask = mask.unsqueeze(0).to(device)
            
            with torch.no_grad():
                scores = model.get_anomaly_score(particles, mask)
            
            results.append({
                'total_score': scores['total_score'].item(),
                'recon_error': scores['recon_error'].item(),
                'is_anomaly': scores['total_score'].item() > ANOMALY_THRESHOLD
            })
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )