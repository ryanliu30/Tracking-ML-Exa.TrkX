import sys

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint

from ..gnn_base import GNNBase
from ..utils import make_mlp

class VanillaCheckResAGNN(GNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        concatenation_factor = 3 if (self.hparams["aggregation"] in ["sum_max", "mean_max"]) else 2
        
        self.edge_network = make_mlp(
            (hparams["hidden"]) * 2,
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            hidden_activation=hparams["hidden_activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"],
        )
        
        self.node_network = make_mlp(
            (hparams["hidden"]) * concatenation_factor,
            [hparams["hidden"]] * hparams["nb_node_layer"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=None,
            layer_norm=hparams["layernorm"],
        )
        
        self.input_network = make_mlp(
            hparams["spatial_channels"] + hparams["cell_channels"], 
            [hparams["hidden"]]*hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"]
        )
            
    def forward(self, x, edge_index):
        start, end = edge_index

        x = self.input_network(x)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):
            # x_inital = x

            # Apply edge network
            edge_inputs = torch.cat([x[start], x[end]], dim=1)
            e = checkpoint(self.edge_network, edge_inputs)
            e = torch.sigmoid(e)

            if self.hparams["aggregation"] == "sum":  
                messages = scatter_add(e * x[start], end, dim=0, dim_size=x.shape[0]) 
                                    
            elif self.hparams["aggregation"] == "max":
                messages = scatter_max(e * x[start], end, dim=0, dim_size=x.shape[0])[0]
                
            elif self.hparams["aggregation"] == "sum_max":
                messages = torch.cat([scatter_max(e * x[start], end, dim=0, dim_size=x.shape[0])[0],
                                     scatter_add(e * x[start], end, dim=0, dim_size=x.shape[0])], dim=-1)
                
            
            node_inputs = torch.cat([messages, x], dim=1)
            x = checkpoint(self.node_network, node_inputs)

            # Residual connection
            x = x_inital + x
    
        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        return checkpoint(self.edge_network, edge_inputs)
