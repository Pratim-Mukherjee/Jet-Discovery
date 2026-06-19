import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from .base_ae import BaseAutoencoder

class GraphAutoencoder(BaseAutoencoder):
    """
    Graph Neural Network Autoencoder for particle clouds.
    Treats each particle as a node with features (E, px, py, pz).
    """
    
    def __init__(self, n_particles, latent_dim=32, hidden_dim=64):
        super().__init__(n_particles * 4, latent_dim)
        self.n_particles = n_particles
        self.hidden_dim = hidden_dim
        self.name = "Graph_AE"
        
        # Encoder: Graph Convolutional layers
        self.conv1 = GCNConv(4, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Global pooling to get graph-level representation
        self.fc_pool = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: MLP that expands latent to node features
        self.fc_decode = nn.Linear(latent_dim, n_particles * 4)
    
    def encode(self, x):
        """
        x: [batch, n_particles, 4]
        Returns: [batch, latent_dim]
        """
        batch_size, n_particles, _ = x.shape
        
        # Build graph for each sample in batch
        # We'll process each sample independently
        all_embeddings = []
        
        for b in range(batch_size):
            # Node features: [n_particles, 4]
            node_feats = x[b]
            
            # Create edges: fully connected (or k-nearest in practice)
            # For simplicity, we use a fully connected graph
            # In practice, you'd use distance-based edges in eta-phi space
            edge_index = self._create_edges(n_particles, device=x.device)
            
            # Apply GCN
            h = F.relu(self.conv1(node_feats, edge_index))
            h = F.relu(self.conv2(h, edge_index))
            h = F.relu(self.conv3(h, edge_index))
            
            # Global pooling
            h_graph = global_mean_pool(h, batch=torch.zeros(n_particles, dtype=torch.long, device=x.device))
            
            # Project to latent
            z = self.fc_pool(h_graph)
            all_embeddings.append(z)
        
        return torch.cat(all_embeddings, dim=0)
    
    def _create_edges(self, n, device):
        """Create fully connected edges (excluding self-loops)"""
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
        return edge_index
    
    def decode(self, z):
        """
        z: [batch, latent_dim]
        Returns: [batch, n_particles, 4]
        """
        h = self.fc_decode(z)  # [batch, n_particles * 4]
        return h.view(-1, self.n_particles, 4)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z