import sys

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min 
from torch.utils.checkpoint import checkpoint
from cugraph.structure.symmetrize import symmetrize
from cuml.cluster import HDBSCAN, DBSCAN, KMeans
import cudf
import cupy as cp

from .gnn_utils import InteractionGNNCell, HierarchicalGNNCell, DynamicGraphConstruction
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
    
class HierarchicalGNNBlock(nn.Module):

    """
    An interaction network class
    """

    def __init__(self, hparams, logging):
        super().__init__()
            
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """                
        
        self.supernode_encoder = make_mlp(
            hparams["latent"],
            hparams["hidden"],
            hparams["latent"] - hparams["emb_dim"],
            hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.superedge_encoder = make_mlp(
            2 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # Initialize GNN blocks
        if hparams["share_weight"]:
            cell = HierarchicalGNNCell(hparams)
            hgnn_cells = [
                cell
                for _ in range(hparams["n_hierarchical_graph_iters"])
            ]
        else:
            hgnn_cells = [
                HierarchicalGNNCell(hparams)
                for _ in range(hparams["n_hierarchical_graph_iters"])
            ]
        
        self.hgnn_cells = nn.ModuleList(hgnn_cells)
        
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
        
        self.hdbscan_model = HDBSCAN(min_cluster_size = hparams["min_cluster_size"])
        self.super_graph_construction = DynamicGraphConstruction("sigmoid", hparams)
        self.bipartite_graph_construction = DynamicGraphConstruction("exp", hparams)
        
        self.log = logging
        self.hparams = hparams

        
    def forward(self, x, embeddings, nodes, edges, graph):
        
        x.requires_grad = True
        
        with torch.no_grad():
            
            clustering_input = embeddings + torch.normal(0, 1e-3, embeddings.shape, device = embeddings.device)
                
            clustering_input = cudf.DataFrame(clustering_input.detach())       
            clusters = self.hdbscan_model.fit_predict(clustering_input)

            del clustering_input

            clusters = torch.tensor(clusters, device = embeddings.device).long()
            if (clusters >= 0).any():
                clusters[clusters >= 0] = clusters[clusters >= 0].unique(return_inverse = True)[1]
            if (clusters < 0).all():
                clusters = clusters + 1

        # Compute Centers
        means = scatter_mean(embeddings[clusters >= 0], clusters[clusters >= 0], dim=0, dim_size=clusters.max()+1)
        means = nn.functional.normalize(means)
        
        super_graph, super_edge_weights = self.super_graph_construction(means, means, sym = True, norm = False, k = self.hparams["supergraph_sparsity"])
        bipartite_graph, bipartite_edge_weights = self.bipartite_graph_construction(embeddings, means, sym = False, norm = True, k = self.hparams["bipartitegraph_sparsity"])
        
        self.log("clusters", len(means))
        
        supernodes = scatter_add((nodes[bipartite_graph[0]])*bipartite_edge_weights, bipartite_graph[1], dim=0, dim_size=means.shape[0])
        supernodes = torch.cat([means, checkpoint(self.supernode_encoder, supernodes)], dim = -1)
        superedges = checkpoint(self.superedge_encoder, torch.cat([supernodes[super_graph[0]], supernodes[super_graph[1]]], dim=1))
        
        del means, embeddings
        
        for layer in self.hgnn_cells:
            nodes, edges, supernodes, superedges = layer(nodes,
                                                         edges,
                                                         supernodes,
                                                         superedges,
                                                         graph,
                                                         bipartite_graph,
                                                         bipartite_edge_weights,
                                                         super_graph,
                                                         super_edge_weights)
            
        embeddings = self.output_layer(nodes)
        embeddings = nn.functional.normalize(embeddings) 
        
        return embeddings, clusters
    
class  Embedding_HierarchicalGNN_HDNSCAN(EmbeddingBase):

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