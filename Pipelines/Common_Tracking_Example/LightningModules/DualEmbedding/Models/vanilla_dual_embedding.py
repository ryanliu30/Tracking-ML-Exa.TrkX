# System imports
import sys
import os
import copy

# 3rd party imports
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
import torch
from ..dual_embedding_base import DualEmbeddingBase


class VanillaDualEmbedding(DualEmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """

        # Construct the MLP architecture
        in_channels = (hparams["spatial_channels"] + hparams["cell_channels"])
        
        self.input_layer1 = Linear(in_channels, hparams["hidden"])
        
        
        layers1 = [
            Linear(hparams["hidden"], hparams["hidden"])
            for _ in range(hparams["nb_layer"] - 1)
        ]

        
        self.layers1 = nn.ModuleList(layers1)

        
        self.output_layer1 = nn.Linear(hparams["hidden"], hparams["emb_dim"])
        if self.hparams["use_dual_encoder"]:
            self.input_layer2 = Linear(in_channels, hparams["hidden"])
            layers2 = [
                Linear(hparams["hidden"], hparams["hidden"])
                for _ in range(hparams["nb_layer"] - 1)
            ]
            self.layers2 = nn.ModuleList(layers2)
            self.output_layer2 = nn.Linear(hparams["hidden"], hparams["emb_dim"])
        
        self.act = nn.GELU()

    def forward(self, x):
        
        x1 = self.input_layer1(x)

        for l in self.layers1:
            x1 = l(x1)
            x1 = self.act(x1)

        x1 = self.output_layer1(x1)
        x1 = nn.functional.normalize(x1, p=2.0, dim=1, eps=1e-12)
        if self.hparams["use_dual_encoder"]:
            x2 = self.input_layer2(x)

            for l in self.layers2:
                x2 = l(x2)
                x2 = self.act(x2)
                
            x2 = self.output_layer1(x2)
            x2 = nn.functional.normalize(x2, p=2.0, dim=1, eps=1e-12)
        else:
            x2 = x1
        
        return x1, x2
