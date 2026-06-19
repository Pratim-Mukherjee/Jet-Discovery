import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tables
from pathlib import Path
from typing import Optional, Tuple

class JetDataset(Dataset):
    """Top Quark Tagging Dataset – handles HDF5 block format and slices extra columns."""
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = "train",
        max_particles: int = 200,
        normalize: bool = True,
        use_labels: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_particles = max_particles
        self.normalize = normalize
        self.use_labels = use_labels
        
        self.h5_path = self.data_dir / f"{split}.h5"
        if not self.h5_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.h5_path}")
        
        self._load_data()
        if normalize and split == "train":
            self._compute_stats()
    
    def _load_data(self):
        with tables.open_file(self.h5_path, mode='r') as h5file:
            # Find the Table node
            table_node = None
            for node in h5file.walk_nodes():
                if isinstance(node, tables.Table):
                    table_node = node
                    break
            if table_node is None:
                raise ValueError("No Table node found.")
            
            print(f"📋 Using table node: {table_node._v_pathname}")
            print(f"📋 Table columns: {table_node.colnames}")
            
            data = table_node.read()
            field_names = data.dtype.names
            print(f"📋 Fields: {field_names}")
            
            # Look for values_block_* columns
            block_cols = [f for f in field_names if f.startswith('values_block_')]
            if block_cols:
                blocks = [data[col] for col in block_cols]
                combined = np.concatenate(blocks, axis=1)
                print(f"📊 Combined blocks shape: {combined.shape}")
                
                expected = self.max_particles * 4  # 800
                if combined.shape[1] == expected:
                    particles = combined.reshape(-1, self.max_particles, 4)
                elif combined.shape[1] > expected:
                    print(f"⚠️ Combined features = {combined.shape[1]}, taking first {expected} columns.")
                    combined = combined[:, :expected]
                    particles = combined.reshape(-1, self.max_particles, 4)
                else:
                    raise ValueError(f"Combined features = {combined.shape[1]}, expected at least {expected}")
                self.particles = particles.astype(np.float32)
                print(f"✅ Reshaped to {self.particles.shape}")
            else:
                raise ValueError("No values_block_* columns found.")
            
            # Labels (optional)
            self.labels = None
            if self.use_labels:
                if 'type' in field_names:
                    self.labels = data['type'].flatten()
                else:
                    for node_name in ['/labels', '/y', '/label']:
                        try:
                            label_node = h5file.get_node(node_name)
                            self.labels = label_node.read().flatten()
                            break
                        except:
                            pass
                if self.labels is not None:
                    print(f"✅ Extracted labels, shape: {self.labels.shape}")
    
    def _compute_stats(self):
        flat = self.particles.reshape(-1, 4)
        mask = (flat != 0).any(axis=1)
        valid = flat[mask] if np.any(mask) else flat
        self.mean = np.mean(valid, axis=0)
        self.std = np.std(valid, axis=0) + 1e-8
        print(f"📊 Normalization stats: mean={self.mean}, std={self.std}")
    
    def __len__(self):
        return len(self.particles)
    
    def __getitem__(self, idx):
        particles = self.particles[idx].astype(np.float32)
        particles = torch.tensor(particles, dtype=torch.float32)
        mask = torch.any(particles != 0, dim=1).float()
        if self.normalize and hasattr(self, 'mean'):
            particles = (particles - torch.tensor(self.mean, dtype=torch.float32)) / torch.tensor(self.std, dtype=torch.float32)
        
        # Jet-level features for physics loss
        E = torch.sum(particles[:, 0] * mask)
        px = torch.sum(particles[:, 1] * mask)
        py = torch.sum(particles[:, 2] * mask)
        pz = torch.sum(particles[:, 3] * mask)
        mass2 = E**2 - px**2 - py**2 - pz**2
        mass = torch.sqrt(torch.clamp(mass2, min=0))
        pt = torch.sqrt(px**2 + py**2)
        jet_features = torch.tensor([E, px, py, pz, mass, pt], dtype=torch.float32)
        
        if self.use_labels and self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return particles, mask, jet_features, label
        return particles, mask, jet_features


def get_dataloaders(
    data_dir: str,
    batch_size: int = 256,
    max_particles: int = 200,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = JetDataset(data_dir, "train", max_particles, normalize=True, use_labels=False)
    val_dataset = JetDataset(data_dir, "val", max_particles, normalize=True, use_labels=False)
    test_dataset = JetDataset(data_dir, "test", max_particles, normalize=True, use_labels=False)
    
    if hasattr(train_dataset, 'mean'):
        val_dataset.mean = train_dataset.mean
        val_dataset.std = train_dataset.std
        test_dataset.mean = train_dataset.mean
        test_dataset.std = train_dataset.std
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader