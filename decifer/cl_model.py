#!/usr/bin/env python3
import torch.nn as nn

# Model definitions
class CLEncoder(nn.Module):
    def __init__(
        self, 
        input_dim: int = 1000,
        hidden_embedding_dim: int = 512,
        embedding_dim: int = 512, 
        hidden_proj_dim: int = 1024,
        proj_dim: int = 32, 
    ):
        super(CLEncoder, self).__init__()

        # Encoder head
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_embedding_dim),
            nn.ReLU(),
            nn.Linear(hidden_embedding_dim, embedding_dim)
        )
        
        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_proj_dim),
            nn.ReLU(),
            nn.Linear(hidden_proj_dim, proj_dim)
        )

    def forward(self, aug1, aug2):
        # Extract encoder embeddings
        h_1 = self.enc(aug1)
        h_2 = self.enc(aug2)

        # Extract low-dimensional embeddings
        h_1_latent = self.proj(h_1)
        h_2_latent = self.proj(h_2)

        return h_1, h_2, h_1_latent, h_2_latent
