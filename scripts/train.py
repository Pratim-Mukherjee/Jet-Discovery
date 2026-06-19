#!/usr/bin/env python
"""
Train the Hybrid Physics-Aware Ensemble Autoencoder.
Use --quick for 10% data, --tiny for only 1% data (very fast).
"""
import sys
import traceback
from pathlib import Path

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
    
    # Parse flags
    tiny_mode = "--tiny" in sys.argv
    quick_mode = "--quick" in sys.argv
    
    if tiny_mode:
        print("⚡⚡ TINY MODE ENABLED (ultra-fast)")
        print("   - Using 10,000 training samples (1%)")
        print("   - max_particles = 50")
        print("   - latent_dim = 8")
        print("   - epochs = 2")
        print("   (Training will finish in ~5-10 minutes on CPU)\n")
    elif quick_mode:
        print("⚡ QUICK MODE ENABLED")
        print("   - Using 120,000 training samples (10%)")
        print("   - max_particles = 100")
        print("   - latent_dim = 16")
        print("   - epochs = 3")
        print("   (Training will take ~1-2 hours on CPU)\n")
    else:
        print("ℹ️  Full mode (default) uses all data – may take days on CPU.")
        print("   Use --quick or --tiny for faster training.\n")
    
    # Load config
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides based on mode
    if tiny_mode:
        config['data']['max_particles'] = 50
        config['model']['latent_dim'] = 8
        config['training']['epochs'] = 2
        max_samples_train = 10000      # Only 1% of 1.2M
        max_samples_val = 5000         # optional: also limit val
        max_samples_test = 5000
    elif quick_mode:
        config['data']['max_particles'] = 100
        config['model']['latent_dim'] = 16
        config['training']['epochs'] = 3
        max_samples_train = 120000     # 10%
        max_samples_val = None
        max_samples_test = None
    else:
        max_samples_train = None
        max_samples_val = None
        max_samples_test = None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=project_root / "data",
        batch_size=config['training']['batch_size'],
        max_particles=config['data']['max_particles'],
        num_workers=0,
        max_samples_train=max_samples_train,
        max_samples_val=max_samples_val,
        max_samples_test=max_samples_test
    )
    print(f"✓ Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
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
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
    
    epochs = config['training']['epochs']
    print(f"\nStarting training for {epochs} epochs...")
    history = trainer.train(epochs=epochs, save_dir=project_root / "checkpoints")
    
    print(f"\n✅ Training complete! Best val loss: {min(history['val_loss']):.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted.")
        sys.exit(1)
    except Exception as e:
        print("\n❌ ERROR:")
        traceback.print_exc()
        sys.exit(1)