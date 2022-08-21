import sys

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch.utils.checkpoint import checkpoint

from .gnn_utils import InteractionGNNCell
from ..embedding_base import EmbeddingBase
from ..utils import make_mlp, find_neighbors
    
class InteractionGNNBlock(nn.Module):

    """
    An interaction network class
    """

    def __init__(self, hparams, iterations):
        super().__init__()
            
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """                 
        
        # Setup input network
        self.node_encoder = make_mlp(
            hparams["spatial_channels"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["spatial_channels"]),
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # Initialize GNN blocks
        if hparams["share_weight"]:
            cell = InteractionGNNCell(hparams)
            ignn_cells = [
                cell
                for _ in range(iterations)
            ]
        else:
            ignn_cells = [
                InteractionGNNCell(hparams)
                for _ in range(iterations)
            ]
        
        self.ignn_cells = nn.ModuleList(ignn_cells)
        
        # output layers
        self.output_layer = make_mlp(
            hparams["latent"],
            hparams["hidden"],
            hparams["emb_dim"],
            hparams["output_layers"],
            layer_norm=hparams["layernorm"],
            output_activation= None,
            hidden_activation=hparams["hidden_output_activation"],
        )
        
        self.hparams = hparams
        
    def forward(self, x, graph):
        
        x.requires_grad = True
        
        nodes = checkpoint(self.node_encoder, x)
        edges = checkpoint(self.edge_encoder, torch.cat([x[graph[0]], x[graph[1]]], dim=1))
        
        for layer in self.ignn_cells:
            nodes, edges= layer(nodes, edges, graph)
        
        embeddings = self.output_layer(nodes)
        embeddings = nn.functional.normalize(embeddings) 
        
        return embeddings, nodes, edges
    

    
class HierarchicalGNN(EmbeddingBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams) 
        
        if hparams["checkpointing"]:
            from torch.utils.checkpoint import checkpoint
        else:
            global checkpoint
            checkpoint = lambda i, *j: i(*j)

        
        self.ignn_block = InteractionGNNBlock(hparams, hparams["n_interaction_graph_iters"])
        self.hgnn_block = HierarchicalGNNBlock(hparams, self.log)
        
    def forward(self, x, graph):
        
        x.requires_grad = True
        
        directed_graph = torch.cat([graph, graph.flip(0)], dim = 1)
        
        intermediate_embeddings, nodes, edges = self.ignn_block(x, directed_graph)
        
        embeddings, clusters = self.hgnn_block(x, intermediate_embeddings, nodes, edges, directed_graph)       
        
        return (embeddings, intermediate_embeddings, clusters)

class Embedding_InteractionGNN(EmbeddingBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams) 
        
        if hparams["checkpointing"]:
            from torch.utils.checkpoint import checkpoint
        else:
            global checkpoint
            checkpoint = lambda i, *j: i(*j)

        
        self.ignn_block = InteractionGNNBlock(hparams, hparams["n_pure_interaction_graph_iters"])
        
    def forward(self, x, graph):
        
        x.requires_grad = True
        
        directed_graph = torch.cat([graph, graph.flip(0)], dim = 1)
        
        embeddings, nodes, edges = self.ignn_block(x, directed_graph)      
        
        return (embeddings, )
