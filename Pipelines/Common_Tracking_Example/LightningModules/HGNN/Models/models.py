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

from ..object_condensation_base import ObjectCondensationBase
from ..utils import make_mlp, find_neighbors

class InteractionGNNBlock(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        if hparams["use_toy"]:
            hparams["regime"] = []
            hparams["spatial_channels"] = 2
        
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
        
    def forward(self, nodes, edges, graph, edge_weights):
        
        # Compute new node features
        edge_messages = scatter_add(edges*edge_weights, graph[1], dim=0, dim_size=nodes.shape[0])
        node_input = torch.cat([nodes, edge_messages], dim=-1)
        nodes = checkpoint(self.node_network, node_input) + nodes
        del node_input, edge_messages
            
        # Compute new edge features
        edge_input = torch.cat([nodes[graph[0]], nodes[graph[1]], edges], dim=-1)
        edges = checkpoint(self.edge_network, edge_input) + edges
        del edge_input
        
        return nodes, edges

class HierarchicalGNNBlock(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        in_channels = hparams["spatial_channels"]
        if "ci" in hparams["regime"]:
            in_channels = in_channels + hparams["cell_channels"]    
        
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
        
    def forward(self, nodes, edges, supernodes, superedges, graph, edge_weights, bipartite_graph, bipartite_edge_weights, super_graph, super_edge_weights):
        
        # Compute new supernode features
        node_messages = scatter_add(bipartite_edge_weights*nodes[bipartite_graph[0]], bipartite_graph[1], dim=0, dim_size=bipartite_graph[1].max()+1)
        attention_messages = scatter_add(superedges[super_graph[0]]*super_edge_weights, super_graph[1], dim=0, dim_size=supernodes.shape[0])
        supernodes = checkpoint(self.supernode_network, torch.cat([supernodes, attention_messages, node_messages], dim=-1)) + supernodes
        del node_messages, attention_messages
        
        # Compute original graph updates
        supernode_messages = scatter_add(bipartite_edge_weights*supernodes[bipartite_graph[1]], bipartite_graph[0], dim=0, dim_size=nodes.shape[0])
        edge_messages = scatter_add(edges*edge_weights, graph[1], dim=0, dim_size=nodes.shape[0])
        nodes = checkpoint(self.node_network, torch.cat([nodes, edge_messages, supernode_messages], dim=-1)) + nodes
        del supernode_messages, edge_messages
        
        # Compute new superedge features
        superedges = checkpoint(self.superedge_network, torch.cat([supernodes[super_graph[0]], supernodes[super_graph[1]], superedges], dim=-1))+ superedges
        
        # Compute new edge features
        edges = checkpoint(self.edge_network, torch.cat([nodes[graph[0]], nodes[graph[1]], edges], dim=-1)) + edges
        
        return nodes, edges, supernodes, superedges

class DynamicGraphConstruction(nn.Module):
    def __init__(self, weighting_function, hparams):
        super().__init__()
        
        self.weight_normalization = nn.BatchNorm1d(1)  
        self.weighting_function = getattr(torch, weighting_function)
    
    def forward(self, src_embeddings, dst_embeddings, sym, norm, k, r):
        
        # Construct the Graph
        with torch.no_grad():
            graph_idxs = find_neighbors(src_embeddings, dst_embeddings, r_max=r, k_max=k)
            positive_idxs = (graph_idxs >= 0)
            ind = torch.arange(graph_idxs.shape[0], device = src_embeddings.device).unsqueeze(1).expand(graph_idxs.shape)
            if sym:
                src, dst = symmetrize(cudf.Series(ind[positive_idxs]), cudf.Series(graph_idxs[positive_idxs]))
                graph = torch.tensor(cp.vstack([src.to_cupy(), dst.to_cupy()]), device=src_embeddings.device).long()
            else:
                src, dst = ind[positive_idxs], graph_idxs[positive_idxs]
                graph = torch.stack([src, dst], dim = 0)
        
        # Compute bipartite attention
        likelihood = torch.einsum('ij,ij->i', src_embeddings[graph[0]], dst_embeddings[graph[1]]) 
        edge_weights_logits = self.weight_normalization(likelihood.unsqueeze(1)).squeeze()
        edge_weights = self.weighting_function(edge_weights_logits)
        if norm:
            edge_weights = edge_weights/(1e-12 + scatter_add(edge_weights, graph[0], dim=0, dim_size = src_embeddings.shape[0])[graph[0]])
        edge_weights = edge_weights.unsqueeze(1)
        
        return graph, edge_weights

class GraphConstruction(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        in_channels = hparams["spatial_channels"]
        if "ci" in hparams["regime"]:
            in_channels = in_channels + hparams["cell_channels"]  
        
        self.embedding_network = make_mlp(
            in_channels,
            hparams["mlp_hidden"],
            hparams["emb_dim"],
            hparams["nb_mlp_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.graph_construction = DynamicGraphConstruction("sigmoid", hparams)
        
        self.hparams = hparams
        
    def forward(self, x):
        
        embeddings = checkpoint(self.embedding_network, x)
        
        embeddings = nn.functional.normalize(embeddings) 
        
        graph, edge_weights = self.graph_construction(embeddings, embeddings, True, False, self.hparams["graph_sparsity"], 1.0)
        
        return embeddings, graph, edge_weights
    
class SuperGraphConstruction(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        in_channels = hparams["spatial_channels"]
        if "ci" in hparams["regime"]:
            in_channels = in_channels + hparams["cell_channels"]  
        
        self.clustering_network = make_mlp(
            hparams["latent"],
            hparams["hidden"],
            hparams["emb_dim"],
            hparams["output_layers"],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_output_activation"],
        )
        
        self.superedge_encoder = make_mlp(
            2*hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.supernode_encoder = make_mlp(
            hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.super_graph_construction =  DynamicGraphConstruction("sigmoid", hparams)
        self.bipartite_graph_construction = DynamicGraphConstruction("exp", hparams)
        
        self.model = HDBSCAN(min_cluster_size = hparams["min_cluster_size"])
        
        self.hparams = hparams
        
    def forward(self, embedded_nodes, encoded_nodes): 
        
        # Compute Embeddings
        embeddings = self.clustering_network(embedded_nodes)
        embeddings = nn.functional.normalize(embeddings)
        
        with torch.no_grad():
            clustering_input = embeddings * torch.normal(1, 5e-3, embeddings.shape[:1], device = embeddings.device).unsqueeze(1)
            # clustering_input = cudf.DataFrame(clustering_input.detach())       
            clusters = self.model.fit_predict(clustering_input)

            del clustering_input

            clusters = torch.tensor(clusters, device = embedded_nodes.device).long()
            if (clusters >= 0).any():
                clusters[clusters >= 0] = clusters[clusters >= 0].unique(return_inverse = True)[1]
            if (clusters < 0).all():
                clusters = clusters + 1

        # Compute Centers
        means = scatter_mean(embeddings[clusters>=0], clusters[clusters>=0], dim=0, dim_size=clusters.max()+1)
        means = nn.functional.normalize(means)
        
        # Construct Graphs
        super_graph, super_edge_weights = self.super_graph_construction(means, means, True, False, self.hparams["supergraph_sparsity"], 1.0)
        bipartite_graph, bipartite_edge_weights = self.bipartite_graph_construction(
            embeddings, means, False, True, self.hparams["bipartitegraph_sparsity"], 1.0)
        
        # Aggregate Supernode Features
        node_messages = checkpoint(self.supernode_encoder, encoded_nodes) 
        supernodes = scatter_add((node_messages[bipartite_graph[0]])*bipartite_edge_weights, bipartite_graph[1], dim=0, dim_size=bipartite_graph[1].max()+1)
        superedges = checkpoint(self.superedge_encoder, torch.cat([supernodes[super_graph[0]], supernodes[super_graph[1]]], dim=1))
        
        return embeddings, supernodes, superedges, bipartite_graph, bipartite_edge_weights, super_graph, super_edge_weights
    
class HierarchicalGNN(ObjectCondensationBase):

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
            
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        in_channels = hparams["spatial_channels"]
        if "ci" in hparams["regime"]:
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
        
        self.bipartite_edge_scoring = make_mlp(
            2*hparams["latent"],
            hparams["hidden"],
            1,
            hparams["output_layers"],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation="Tanh",
        )
        
        self.graph_construction = GraphConstruction(hparams)
        self.supergraph_construction = SuperGraphConstruction(hparams)

        # Initialize GNN blocks
        if hparams["share_weight"]:
            block = InteractionGNNBlock(hparams)
            ignn_blocks = [
                block
                for _ in range(self.hparams["n_interaction_graph_iters"])
            ]
        else:
            ignn_blocks = [
                InteractionGNNBlock(hparams)
                for _ in range(self.hparams["n_interaction_graph_iters"])
            ]
            
        if hparams["share_weight"]:
            block = HierarchicalGNNBlock(hparams)
            hgnn_blocks = [
                block
                for _ in range(self.hparams["n_hierarchical_graph_iters"])
            ]
        else:
            hgnn_blocks = [
                HierarchicalGNNBlock(hparams)
                for _ in range(self.hparams["n_hierarchical_graph_iters"])
            ]
        
        self.ignn_blocks = nn.ModuleList(ignn_blocks)
        self.hgnn_blocks = nn.ModuleList(hgnn_blocks)
        
        # output layers
        self.output_layer = make_mlp(
            hparams["latent"],
            hparams["hidden"],
            1,
            hparams["output_layers"],
            layer_norm=hparams["layernorm"],
            output_activation= None,
            hidden_activation=hparams["hidden_output_activation"],
        )
        
        
        self.softmax = nn.Softmax(dim = -1)

        
    def forward(self, x, cell_info):
    
        x.requires_grad = True
        _, directed_graph, edge_weights = self.graph_construction(x)
        
        nodes = checkpoint(self.node_encoder, x)
        edges = checkpoint(self.edge_encoder, torch.cat([x[directed_graph[0]], x[directed_graph[1]]], dim=1))
        
        for layer in self.ignn_blocks:
            nodes, edges= layer(nodes, edges, directed_graph, edge_weights)
            
        intermediate_embeddings, supernodes, superedges, bipartite_graph, bipartite_edge_weights, super_graph, super_edge_weights = \
                    checkpoint(self.supergraph_construction, nodes.clone(), nodes.clone())
        
        self.log("clusters", float(len(supernodes)))
        
        for layer in self.hgnn_blocks:
            nodes, edges, supernodes, superedges = layer(nodes,
                                                         edges,
                                                         supernodes,
                                                         superedges,
                                                         directed_graph,
                                                         edge_weights,
                                                         bipartite_graph,
                                                         bipartite_edge_weights,
                                                         super_graph,
                                                         super_edge_weights)
            
        bipartite_edge_scores = checkpoint(self.bipartite_edge_scoring,
                                           torch.cat([nodes[bipartite_graph[0]], supernodes[bipartite_graph[1]]], dim=-1)).squeeze()
        bipartite_edge_scores = torch.sigmoid(bipartite_edge_scores)
        supernodes = checkpoint(self.output_layer, supernodes)
        
        return intermediate_embeddings, bipartite_edge_scores, bipartite_graph, supernodes.squeeze()