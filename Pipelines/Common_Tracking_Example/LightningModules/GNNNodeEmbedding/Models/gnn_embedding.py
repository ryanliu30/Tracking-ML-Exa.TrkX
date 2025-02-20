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
        
        if hparams["use_toy"]:
            hparams["regime"] = []
            hparams["spatial_channels"] = 2
            
        global_size = 0
        if hparams["global_information"]:
            global_size = hparams["hidden"]
            self.global_node_aggregator = make_mlp(
                hparams["latent"],
                hparams["hidden"],
                hparams["latent"],
                hparams["nb_node_layer"],
                layer_norm=hparams["layernorm"],
                output_activation="Tanh",
                hidden_activation=hparams["hidden_activation"],
            )
            
            self.global_edge_aggregator = make_mlp(
                hparams["latent"],
                hparams["hidden"],
                hparams["latent"],
                hparams["nb_edge_layer"],
                layer_norm=hparams["layernorm"],
                output_activation="Tanh",
                hidden_activation=hparams["hidden_activation"],
            )
        
        self.edge_network = make_mlp(
            3 * hparams["latent"] + global_size,
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation="Tanh",
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            2 * hparams["latent"] + global_size,
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.hparams = hparams
        
    def forward(self, nodes, edges, graph):
        
        nodes_res = nodes.clone()
        edges_res = edges.clone()
        
        # Compute new node features
        edge_messages = scatter_add(edges, graph[1], dim=0, dim_size=nodes.shape[0])
        
        if self.hparams["global_information"]:
            global_node_information = checkpoint(self.global_node_aggregator, nodes_res)
            node_inputs = torch.cat([nodes, edge_messages, global_node_information.mean(0).unsqueeze(0).expand(nodes_res.shape)], dim=-1)
        else:
            node_inputs = torch.cat([nodes, edge_messages], dim=-1)
            
        nodes = checkpoint(self.node_network, node_inputs) + nodes_res
            
        # Compute new edge features
        if self.hparams["global_information"]:
            global_edge_information = checkpoint(self.global_edge_aggregator, edges_res)
            edge_input = torch.cat([nodes[graph[0]], nodes[graph[1]], edges, global_edge_information.mean(0).unsqueeze(0).expand(edges_res.shape)], dim=-1)
        else:
            edge_input = torch.cat([nodes[graph[0]], nodes[graph[1]], edges], dim=-1)

        edges = checkpoint(self.edge_network, edge_input) + edges_res
        
        return nodes, edges

class InteractionGNN(GNNEmbeddingBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        if hparams["use_toy"]:
            hparams["regime"] = []
            hparams["spatial_channels"] = 2
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
            hparams["latent"],
            hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            (2 * in_channels),
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        
        # Initialize GNN blocks
        if hparams["share_weight"]:
            block = InteractionGNNBlock(hparams)
            gnn_blocks = [
                block
                for _ in range(self.hparams["n_graph_iters"])
            ]
        else:
            gnn_blocks = [
                InteractionGNNBlock(hparams)
                for _ in range(self.hparams["n_graph_iters"])
            ]
            
        
        self.gnn_blocks = nn.ModuleList(gnn_blocks)
        
        # output layers
        self.output_layer = make_mlp(
            hparams["latent"],
            hparams["hidden"],
            hparams["emb_dim"],
            hparams["output_layers"],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_output_activation"],
        )
        

    def forward(self, x, graph):
        
        x.requires_grad = True
        
        directed_graph = torch.cat([graph, graph.flip(0)], dim = 1)
        nodes = checkpoint(self.node_encoder, x)
        edges = checkpoint(self.edge_encoder, torch.cat([x[directed_graph[0]], x[directed_graph[1]]], dim=1))

        for layer in self.gnn_blocks:
            nodes, edges = layer(nodes, edges, directed_graph)
            
        embeddings = self.output_layer(nodes)
        
        return nn.functional.normalize(embeddings, p=2.0, dim=1, eps=1e-12)



        