# System imports
import sys
import os
import copy

# 3rd party imports
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
import torch
from ..edge_embedding_base import EdgeEmbeddingBase


class VanillaEdgeEmbedding(EdgeEmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """

        # Construct the MLP architecture
        if self.hparams["use_difference"]:
            in_channels = hparams["spatial_channels"] + 2*hparams["cell_channels"] + 2
        else:
            in_channels = (hparams["spatial_channels"] + hparams["cell_channels"]) * 2
        
        self.input_layer1 = Linear(in_channels, hparams["hidden"])
        self.input_layer2 = Linear(in_channels, hparams["hidden"])
        
        layers1 = [
            Linear(hparams["hidden"], hparams["hidden"])
            for _ in range(hparams["nb_layer"] - 1)
        ]
        layers2 = [
            Linear(hparams["hidden"], hparams["hidden"])
            for _ in range(hparams["nb_layer"] - 1)
        ]
        
        self.layers1 = nn.ModuleList(layers1)
        self.layers2 = nn.ModuleList(layers2)
        
        self.output_layer1 = nn.Linear(hparams["hidden"], hparams["emb_dim"])
        self.output_layer2 = nn.Linear(hparams["hidden"], hparams["emb_dim"])
        
        self.act = nn.GELU()
        
    def get_difference(self, x1, x2):
        
        ci = self.hparams["cell_channels"]
        
        delta_theta = (x1[:,ci+1] - x2[:,ci+1]).unsqueeze(1)
        delta_theta[delta_theta > 1] = delta_theta[delta_theta > 1] - 2
        delta_theta[delta_theta < -1] = delta_theta[delta_theta < -1] + 2
        
        delta_r = (torch.sqrt(x1[:,ci+0].square() + x2[:,ci+2].square())-
                   torch.sqrt(x1[:,ci+0].square() + x2[:,ci+2].square())
                  ).unsqueeze(1)
        
        delta_eta = (torch.log((torch.sqrt(x2[:,ci+0].square() + x2[:,ci+2].square()) - x2[:,ci+2])/x2[:,ci+0]) - 
                     torch.log((torch.sqrt(x1[:,ci+0].square() + x1[:,ci+2].square()) - x1[:,ci+2])/x1[:,ci+0])
                    ).unsqueeze(1)
        delta_eta = torch.clamp(delta_eta, min = -2, max = 2)/2
        
        base_r = x1[:,ci+0].unsqueeze(1)
        
        base_z = x1[:,ci+2].unsqueeze(1)
        
        new_x = torch.cat([x1[:,:ci],
                           x2[:,:ci],
                           delta_theta,
                           delta_r,
                           delta_eta,
                           base_r, 
                           base_z
                          ], dim = -1)
        return new_x

    def forward(self, x, e):
        
        if self.hparams["use_difference"]:
            x = self.get_difference(x[e[0]], x[e[1]])
        else:
            x = torch.cat([x[e[0]], x[e[1]]], dim = -1)
        
        x1 = self.input_layer1(x)
        x2 = self.input_layer2(x)
        
        for l in self.layers1:
            x1 = l(x1)
            x1 = self.act(x1)
            if self.hparams["layernorm"]:
                x1 = F.layer_norm(x1, (l.out_features,))  # Option of LayerNorm
        for l in self.layers2:
            x2 = l(x2)
            x2 = self.act(x2)
            if self.hparams["layernorm"]:
                x2 = F.layer_norm(x2, (l.out_features,))  # Option of LayerNorm
                
        x1 = self.output_layer1(x1)
        x1 = nn.functional.normalize(x1, p=2.0, dim=1, eps=1e-12)
        
        x2 = self.output_layer1(x2)
        x2 = nn.functional.normalize(x2, p=2.0, dim=1, eps=1e-12)
        
        return x1, x2
