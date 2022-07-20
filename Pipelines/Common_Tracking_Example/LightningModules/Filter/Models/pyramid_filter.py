# System imports
import sys
import os
import copy

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.data import DataLoader

# Local imports
from ..utils import graph_intersection
from ..triplet_filter_base import FilterBase


class PyramidFilter(FilterBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """

        # Construct the MLP architecture
        in_channels = (hparams["spatial_channels"] + hparams["cell_channels"]) * 2
        
        self.input_layer = Linear(in_channels, hparams["hidden"])
        layers = [
            Linear(hparams["hidden"], hparams["hidden"])
            for _ in range(hparams["nb_layer"] - 1)
        ]
        
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hparams["hidden"], 1)
        self.act = nn.GELU()

    def forward(self, x, e):
        
        x = self.input_layer(torch.cat([x[e[0]], x[e[1]]], dim=-1))
        
        for l in self.layers:
            x = l(x)
            x = self.act(x)
            if self.hparams["layernorm"]:
                x = F.layer_norm(x, (l.out_features,))  # Option of LayerNorm
        x = self.output_layer(x)
        return x
