import sys

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add
from torch.utils.checkpoint import checkpoint

from ..gnn_embedding_base import GNNEmbeddingBase
from ..utils import make_mlp

class InteractionGNNBlock(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        self.edge_network = make_mlp(
            4 * hparams["hidden"],
            hparams["hidden"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation="Tanh",
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            4 * hparams["hidden"],
            hparams["hidden"],
            hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation="Tanh",
            hidden_activation=hparams["hidden_activation"],
        )
        
    def forward(self, nodes, edges, graph, nodes_res, edges_res):
        
        nodes = torch.cat([nodes, nodes_res], dim=-1)
        edges = torch.cat([edges, edges_res], dim=-1)
        
        # Compute new node features
        edge_messages = scatter_add(
            edges, graph[0], dim=0, dim_size=nodes.shape[0]
        ) + scatter_add(edges, graph[1], dim=0, dim_size=nodes.shape[0])
        node_inputs = torch.cat([nodes, edge_messages], dim=-1)
        nodes = checkpoint(self.node_network, node_inputs)

        # Compute new edge features
        edge_inputs = torch.cat([nodes[graph[0]], nodes[graph[1]], edges], dim=-1)
        edges = checkpoint(self.edge_network, edge_inputs)
        
        return nodes, edges


class InteractionGNN(GNNEmbeddingBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        in_channels = hparams["spatial_channels"]
        if "ci" in self.hparams["regime"]:
            in_channels = in_channels + hparams["cell_channels"]                  
            
        # Setup input network
        self.node_encoder = make_mlp(
            in_channels,
            hparams["hidden"],
            hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["hidden"]),
            hparams["hidden"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )
        
        # Initialize GNN blocks
        gnn_blocks = [
            InteractionGNNBlock(hparams)
            for _ in range(self.hparams["n_graph_iters"])
        ]
        
        self.gnn_blocks = nn.ModuleList(gnn_blocks)
        
        # output layers
        self.output_layer = make_mlp(
            hparams["hidden"],
            hparams["emb_dim"],
            hparams["output_layers"],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_output_activation"],
        )
        

    def forward(self, x, graph):
        
        nodes = checkpoint(self.node_encoder, x)
        edges = checkpoint(self.edge_encoder, torch.cat([nodes[graph[0]], nodes[graph[1]]], dim=1))
        nodes_res, edges_res = nodes.clone(), edges.clone()

        for layer in self.gnn_blocks:
            nodes_old, edges_old = nodes.clone(), edges.clone()
            nodes, edges = layer(nodes, edges, graph, nodes_res, edges_res)
            nodes_res, edges_res = nodes_old.clone(), edges_old.clone()
            
        nodes = self.output_layer(nodes)
        return nodes



        