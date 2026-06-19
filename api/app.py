from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from typing import List
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

from api.schemas import JetInput, AnomalyResponse, HealthResponse
from api.model_loader import ModelLoader

app = FastAPI(
    title="Hybrid Physics-Aware Ensemble Autoencoder API",
    description="Anomaly detection for jet physics using a hybrid ensemble of autoencoders.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_loader = ModelLoader()
ANOMALY_THRESHOLD = 1.5

@app.on_event("startup")
async def startup_event():
    try:
        model_loader.load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")

# ---------- HTML UI ----------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Jet Anomaly Detector</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #f5f7fa; }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        textarea { width: 100%; height: 150px; font-family: monospace; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #3498db; color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #2980b9; }
        #result { margin-top: 20px; padding: 15px; background: #ecf0f1; border-radius: 4px; white-space: pre-wrap; font-family: monospace; }
        .status { display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
        .anomaly { background: #e74c3c; color: white; }
        .normal { background: #2ecc71; color: white; }
        .example { color: #7f8c8d; font-size: 14px; margin-top: 10px; }
        .example a { color: #3498db; cursor: pointer; text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Hybrid Physics-Aware Ensemble Autoencoder</h1>
        <p>Paste a jet (list of particles) in JSON format:</p>
        <textarea id="input" placeholder='{"particles": [{"E":100,"px":50,"py":30,"pz":10}, ...]}'></textarea>
        <br><br>
        <button onclick="predict()">Detect Anomaly</button>
        <div class="example">Try: <a onclick="document.getElementById('input').value = example;">Load example</a></div>
        <div id="result"></div>
    </div>
    <script>
        const example = JSON.stringify({
            "particles": [
                {"E": 100.0, "px": 50.0, "py": 30.0, "pz": 10.0},
                {"E": 80.0, "px": 40.0, "py": 20.0, "pz": 5.0}
            ]
        }, null, 2);
        function predict() {
            const input = document.getElementById('input').value;
            if (!input.trim()) { alert('Please enter jet data.'); return; }
            try { JSON.parse(input); } catch(e) { alert('Invalid JSON: ' + e.message); return; }
            document.getElementById('result').innerHTML = '⏳ Processing...';
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: input
            })
            .then(res => res.json())
            .then(data => {
                const status = data.is_anomaly ? '⚠️ ANOMALY' : '✅ NORMAL';
                const cls = data.is_anomaly ? 'anomaly' : 'normal';
                document.getElementById('result').innerHTML = `
                    <div><span class="status ${cls}">${status}</span></div>
                    <p><strong>Total Score:</strong> ${data.total_anomaly_score.toFixed(4)} (threshold: ${data.threshold})</p>
                    <p><strong>Reconstruction Error:</strong> ${data.reconstruction_error.toFixed(4)}</p>
                    <p><strong>Latent Score:</strong> ${data.latent_score.toFixed(4)}</p>
                    <p><strong>Physics Score:</strong> ${data.physics_score.toFixed(4)}</p>
                    <p><strong>Ensemble Weights:</strong> MLP ${data.ensemble_weights[0].toFixed(3)}, Conv ${data.ensemble_weights[1].toFixed(3)}, Graph ${data.ensemble_weights[2].toFixed(3)}</p>
                    <hr>
                    <pre>${JSON.stringify(data, null, 2)}</pre>
                `;
            })
            .catch(err => {
                document.getElementById('result').innerHTML = '❌ Error: ' + err.message;
            });
        }
        // Load example on page load
        window.onload = function() {
            document.getElementById('input').value = example;
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def homepage():
    return HTML_PAGE

# ---------- API endpoints ----------
@app.get("/health", response_model=HealthResponse)
async def health_check():
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
            n_particles=50,
            latent_dim=8
        )

@app.post("/predict", response_model=AnomalyResponse)
async def predict_anomaly(jet: JetInput):
    try:
        model = model_loader.get_model()
        device = next(model.parameters()).device
        
        particles, n = jet.to_tensor(model.n_particles)
        particles = particles.unsqueeze(0).to(device)
        
        mask = torch.zeros(model.n_particles)
        mask[:n] = 1.0
        mask = mask.unsqueeze(0).to(device)
        
        with torch.no_grad():
            scores = model.get_anomaly_score(particles, mask)
        
        total_score = scores['total_score'].detach().item()
        recon_error = scores['recon_error'].detach().item()
        latent_score = scores['latent_score'].detach().item()
        physics_score = scores['physics_score'].detach().item()
        recon_mlp = scores['recon_mlp'].detach().item()
        recon_conv = scores['recon_conv'].detach().item()
        recon_graph = scores['recon_graph'].detach().item()
        
        is_anomaly = total_score > ANOMALY_THRESHOLD
        
        weights = scores.get('ensemble_weights')
        if weights is None:
            with torch.no_grad():
                w = torch.softmax(model.fusion_weights, dim=0).cpu().numpy().tolist()
            weights = w
        elif isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy().tolist()
        
        return AnomalyResponse(
            jet_id=None,
            total_anomaly_score=total_score,
            reconstruction_error=recon_error,
            latent_score=latent_score,
            physics_score=physics_score,
            is_anomaly=is_anomaly,
            threshold=ANOMALY_THRESHOLD,
            recon_mlp=recon_mlp,
            recon_conv=recon_conv,
            recon_graph=recon_graph,
            ensemble_weights=weights
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(jets: List[JetInput]):
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
            total_score = scores['total_score'].detach().item()
            results.append({
                'total_score': total_score,
                'recon_error': scores['recon_error'].detach().item(),
                'is_anomaly': total_score > ANOMALY_THRESHOLD
            })
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)