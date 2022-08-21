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

from ..embedding_base import EmbeddingBase
from ..utils import make_mlp, find_neighbors

class InteractionGNNCell(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        self.edge_network = make_mlp(
            3 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation="Tanh",
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            2 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.hparams = hparams
        
    def forward(self, nodes, edges, graph):
        
        # Compute new node features
        edge_messages = scatter_add(edges, graph[1], dim=0, dim_size=nodes.shape[0])
        node_input = torch.cat([nodes, edge_messages], dim=-1)
        nodes = checkpoint(self.node_network, node_input) + nodes
        del node_input, edge_messages
            
        # Compute new edge features
        edge_input = torch.cat([nodes[graph[0]], nodes[graph[1]], edges], dim=-1)
        edges = checkpoint(self.edge_network, edge_input) + edges
        del edge_input
        
        return nodes, edges

class HierarchicalGNNCell(nn.Module):
    def __init__(self, hparams):
        super().__init__()  
        
        self.edge_network = make_mlp(
            3 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation="Tanh",
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            3 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.supernode_network = make_mlp(
            3 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.superedge_network = make_mlp(
            3 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation="Tanh",
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.hparams = hparams
        
    def forward(self, nodes, edges, supernodes, superedges, graph, bipartite_graph, bipartite_edge_weights, super_graph, super_edge_weights):
        
        # Compute new supernode features
        node_messages = scatter_add(bipartite_edge_weights*nodes[bipartite_graph[0]], bipartite_graph[1], dim=0, dim_size=supernodes.shape[0])
        attention_messages = scatter_add(superedges[super_graph[0]]*super_edge_weights, super_graph[1], dim=0, dim_size=supernodes.shape[0])
        supernodes = checkpoint(self.supernode_network, torch.cat([supernodes, attention_messages, node_messages], dim=-1)) + supernodes
        del node_messages, attention_messages
        
        # Compute original graph updates
        supernode_messages = scatter_add(bipartite_edge_weights*supernodes[bipartite_graph[1]], bipartite_graph[0], dim=0, dim_size=nodes.shape[0])
        edge_messages = scatter_add(edges, graph[1], dim=0, dim_size=nodes.shape[0])
        nodes = checkpoint(self.node_network, torch.cat([nodes, edge_messages, supernode_messages], dim=-1)) + nodes
        del supernode_messages, edge_messages
        
        # Compute new superedge features
        superedges = checkpoint(self.superedge_network, torch.cat([supernodes[super_graph[0]], supernodes[super_graph[1]], superedges], dim=-1))+superedges
        
        # Compute new edge features
        edges = checkpoint(self.edge_network, torch.cat([nodes[graph[0]], nodes[graph[1]], edges], dim=-1)) + edges
        
        return nodes, edges, supernodes, superedges

class DynamicGraphConstruction(nn.Module):
    def __init__(self, weighting_function, hparams):
        super().__init__()
        
        self.hparams = hparams
        self.weight_normalization = nn.BatchNorm1d(1)  
        self.weighting_function = getattr(torch, weighting_function)
        self.knn_radius = nn.parameter.Parameter(data=torch.ones(1), requires_grad=False)
    
    def forward(self, src_embeddings, dst_embeddings, sym = False, norm = False, k = 10):
        
        # Construct the Graph
        with torch.no_grad():            
            graph_idxs = find_neighbors(src_embeddings, dst_embeddings, r_max=self.knn_radius, k_max=k)
            positive_idxs = (graph_idxs >= 0)
            ind = torch.arange(graph_idxs.shape[0], device = src_embeddings.device).unsqueeze(1).expand(graph_idxs.shape)
            if sym:
                src, dst = symmetrize(cudf.Series(ind[positive_idxs]), cudf.Series(graph_idxs[positive_idxs]))
                graph = torch.tensor(cp.vstack([src.to_cupy(), dst.to_cupy()]), device=src_embeddings.device).long()
            else:
                src, dst = ind[positive_idxs], graph_idxs[positive_idxs]
                graph = torch.stack([src, dst], dim = 0)
            if self.training:
                maximum_dist = (src_embeddings[graph[0]] - dst_embeddings[graph[1]]).square().sum(-1).sqrt().max()
                self.knn_radius.data = 0.9*self.knn_radius.data + 0.11*maximum_dist
        
        # Compute bipartite attention
        likelihood = torch.einsum('ij,ij->i', src_embeddings[graph[0]], dst_embeddings[graph[1]]) 
        edge_weights_logits = self.weight_normalization(likelihood.unsqueeze(1)).squeeze()
        edge_weights = self.weighting_function(edge_weights_logits)
        if norm:
            edge_weights = edge_weights/(1e-12 + scatter_add(edge_weights, graph[0], dim=0, dim_size = src_embeddings.shape[0])[graph[0]])
        edge_weights = edge_weights.unsqueeze(1)
        
        return graph, edge_weights
    
