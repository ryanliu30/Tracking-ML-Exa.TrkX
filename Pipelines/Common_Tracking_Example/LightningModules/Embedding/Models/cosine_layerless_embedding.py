# System imports
import sys
import os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from ..cosine_embedding_base import CosineEmbeddingBase
from torch.nn import Linear
import torch.nn as nn
from torch_cluster import radius_graph
import torch
from torch_geometric.data import DataLoader

# Local imports
from ..utils import graph_intersection


class CosineLayerlessEmbedding(CosineEmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """

          # Construct the MLP architecture
        if "ci" in hparams["regime"]:
            in_channels = hparams["spatial_channels"] + hparams["cell_channels"]
        else:
            in_channels = hparams["spatial_channels"]
            
        layers = [Linear(in_channels, hparams["emb_hidden"])]
        ln = [
            Linear(hparams["emb_hidden"], hparams["emb_hidden"])
            for _ in range(hparams["nb_shared_layer"] - 1)
        ]
        layers.extend(ln)
    
        
        self.layers = nn.ModuleList(layers)
        
        n_ln = []
        for _ in range(hparams["nb_independent_layer"] - 1):
            n_ln.append(Linear(hparams["emb_hidden"], hparams["emb_hidden"]))
            n_ln.append(nn.GELU())
        n_ln.append(Linear(hparams["emb_hidden"], hparams["emb_dim"]))
    
        n_spaces_layers = [nn.Sequential(*n_ln)
                           for _ in range (hparams["n_spaces"])]
        
        self.n_spaces_layers = nn.ModuleList(n_spaces_layers)
        self.act = nn.GELU()
        
        self.save_hyperparameters()
        
    def normalization(self,x):
        x = x / (torch.sqrt(torch.square(x).sum(-1) + 1).unsqueeze(-1))
        return x

    def forward(self, x):
            
        for l in self.layers:
            x = l(x)
            x = self.act(x)
        
        spatials = torch.stack([self.normalization(i(x)) for i in self.n_spaces_layers], dim = 0)
        
        return spatials