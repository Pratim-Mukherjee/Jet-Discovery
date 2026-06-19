#!/usr/bin/env python
"""
Train the Hybrid Physics-Aware Ensemble Autoencoder.
"""
import sys
import os
import traceback
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml

from data.dataset import get_dataloaders
from models.ensemble import HybridPhysicsEnsembleAE
from training.trainer import Trainer


def main():
    print("=" * 60)
    print("Training Hybrid Physics-Aware Ensemble Autoencoder")
    print("=" * 60)
    
    # 1. Load config
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Config loaded")
    
    # 2. Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    # 3. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=project_root / "data",
        batch_size=config['training']['batch_size'],
        max_particles=config['data']['max_particles'],
        num_workers=0  # Windows-friendly
    )
    print(f"✓ Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # 4. Model
    print("Creating model...")
    model = HybridPhysicsEnsembleAE(
        n_particles=config['data']['max_particles'],
        latent_dim=config['model']['latent_dim'],
        ensemble_weights=config['model']['ensemble_weights'],
        physics_weights={
            'energy_weight': config['physics']['energy_weight'],
            'mass_weight': config['physics']['mass_weight'],
            'pt_weight': config['physics']['pt_weight']
        }
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    
    # 5. Trainer
    lr = float(config['training']['learning_rate'])
    wd = float(config['training']['weight_decay'])
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        weight_decay=wd,
        device=device
    )
    print("✓ Trainer initialized")
    
    # 6. Train
    print("\nStarting training...")
    history = trainer.train(
        epochs=config['training']['epochs'],
        save_dir=project_root / "checkpoints"
    )
    
    print("\n✅ Training complete!")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print("\n❌ ERROR:")
        traceback.print_exc()
        sys.exit(1)