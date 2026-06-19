import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

class Trainer:
    """Training loop for the Hybrid Physics Ensemble AE."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Force conversion to float (in case config reads them as strings)
        lr = float(learning_rate)
        wd = float(weight_decay)
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon': [],
            'val_recon': [],
            'train_physics': [],
            'val_physics': []
        }
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_physics = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            if len(batch) == 4:
                particles, mask, jet_features, labels = batch
            else:
                particles, mask, jet_features = batch
            
            particles = particles.to(self.device)
            mask = mask.to(self.device)
            
            self.optimizer.zero_grad()
            loss_dict = self.model.loss(particles, mask)
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss_dict['total'].item()
            total_recon += loss_dict['recon'].item()
            total_physics += loss_dict['physics'].item()
            
            pbar.set_postfix({
                'loss': loss_dict['total'].item(),
                'recon': loss_dict['recon'].item(),
                'phys': loss_dict['physics'].item()
            })
        
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'recon': total_recon / n_batches,
            'physics': total_physics / n_batches
        }
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_physics = 0
        
        for batch in self.val_loader:
            if len(batch) == 4:
                particles, mask, jet_features, labels = batch
            else:
                particles, mask, jet_features = batch
            
            particles = particles.to(self.device)
            mask = mask.to(self.device)
            
            loss_dict = self.model.loss(particles, mask)
            
            total_loss += loss_dict['total'].item()
            total_recon += loss_dict['recon'].item()
            total_physics += loss_dict['physics'].item()
        
        n_batches = len(self.val_loader)
        return {
            'loss': total_loss / n_batches,
            'recon': total_recon / n_batches,
            'physics': total_physics / n_batches
        }
    
    def train(self, epochs=100, save_dir='checkpoints'):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step(val_metrics['loss'])
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_recon'].append(train_metrics['recon'])
            self.history['val_recon'].append(val_metrics['recon'])
            self.history['train_physics'].append(train_metrics['physics'])
            self.history['val_physics'].append(val_metrics['physics'])
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Recon: {train_metrics['recon']:.4f}, Phys: {train_metrics['physics']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Recon: {val_metrics['recon']:.4f}, Phys: {val_metrics['physics']:.4f}")
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'history': self.history,
                    'val_loss': best_val_loss,
                    'config': {
                        'latent_dim': self.model.latent_dim,
                        'n_particles': self.model.n_particles
                    }
                }, save_dir / 'best_model.pt')
                print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
            
            if epoch % 10 == 0:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'history': self.history,
                    'epoch': epoch
                }, save_dir / f'checkpoint_epoch_{epoch}.pt')
        
        self.plot_history(save_dir)
        return self.history
    
    def plot_history(self, save_dir):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        axes[1].plot(self.history['train_recon'], label='Train')
        axes[1].plot(self.history['val_recon'], label='Val')
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        axes[2].plot(self.history['train_physics'], label='Train')
        axes[2].plot(self.history['val_physics'], label='Val')
        axes[2].set_title('Physics Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_history.png', dpi=150)
        plt.close()