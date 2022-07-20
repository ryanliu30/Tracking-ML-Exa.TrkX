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

class InteractionGNNCell(nn.Module):
    def __init__(self, hparams, hidden_size):
        super().__init__()
        
        self.edge_network = make_mlp(
            3 * hidden_size,
            hidden_size,
            hidden_size,
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation="Tanh",
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            2 * hidden_size,
            hidden_size,
            hidden_size,
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

class HierarchicalGNNCell(nn.Module):
    def __init__(self, hparams, hidden_size):
        super().__init__()  
        
        self.edge_network = make_mlp(
            3 * hidden_size,
            hidden_size,
            hidden_size,
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation="Tanh",
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            3 * hidden_size,
            hidden_size,
            hidden_size,
            hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.supernode_network = make_mlp(
            3 * hidden_size,
            hidden_size,
            hidden_size,
            hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.superedge_network = make_mlp(
            3 * hidden_size,
            hidden_size,
            hidden_size,
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation="Tanh",
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.hparams = hparams
        
    def forward(self, nodes, edges, supernodes, superedges, graph, edge_weights, bipartite_graph, bipartite_edge_weights, super_graph, super_edge_weights):
        
        # Compute new supernode features
        node_messages = scatter_add(bipartite_edge_weights*nodes[bipartite_graph[0]], bipartite_graph[1], dim=0, dim_size=supernodes.shape[0])
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
                self.knn_radius = nn.parameter.Parameter(data = 0.9*self.knn_radius + 0.11*maximum_dist, requires_grad=False)
        
        # Compute bipartite attention
        likelihood = torch.einsum('ij,ij->i', src_embeddings[graph[0]], dst_embeddings[graph[1]]) 
        edge_weights_logits = self.weight_normalization(likelihood.unsqueeze(1)).squeeze()
        edge_weights = self.weighting_function(edge_weights_logits)
        if norm:
            edge_weights = edge_weights/(1e-12 + scatter_add(edge_weights, graph[0], dim=0, dim_size = src_embeddings.shape[0])[graph[0]])
        edge_weights = edge_weights.unsqueeze(1)
        
        return graph, edge_weights
    
class InteractionGNNBlock(ObjectCondensationBase):

    """
    An interaction network class
    """

    def __init__(self, hparams, num_edges, num_iterations, hidden_size):
        super().__init__(hparams)
            
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """                 
            
        # Setup input network
        self.node_encoder = make_mlp(
            hparams["spatial_channels"],
            hidden_size,
            hidden_size - hparams["emb_dim"],
            hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["spatial_channels"]+hparams["emb_dim"]),
            hidden_size,
            hidden_size,
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # Initialize GNN blocks
        if hparams["share_weight"]:
            cell = InteractionGNNCell(hparams, hidden_size)
            ignn_cells = [
                cell
                for _ in range(num_iterations)
            ]
        else:
            ignn_cells = [
                InteractionGNNCell(hparams, hidden_size)
                for _ in range(num_iterations)
            ]
        
        self.ignn_cells = nn.ModuleList(ignn_cells)
        
        # output layers
        self.output_layer = make_mlp(
            hidden_size,
            hidden_size,
            hparams["emb_dim"],
            hparams["output_layers"],
            layer_norm=hparams["layernorm"],
            output_activation= None,
            hidden_activation=hparams["hidden_output_activation"],
        )
        
        self.graph_construction = DynamicGraphConstruction("sigmoid", hparams)
        self.knn_radius = nn.parameter.Parameter(data=torch.ones(1), requires_grad=False)
        self.weight_normalization = nn.BatchNorm1d(1)  
        self.num_edges = num_edges

        
    def forward(self, x, embeddings):
        
        x.requires_grad = True
        
        graph, edge_weights = self.graph_construction(embeddings, embeddings, sym = True, norm = False, k = self.num_edges)
        
        nodes = torch.cat([checkpoint(self.node_encoder, x), embeddings], dim = -1)
        edges = checkpoint(self.edge_encoder, torch.cat([x[graph[0]], embeddings[graph[0]], x[graph[1]], embeddings[graph[1]]], dim=1))
        
        for layer in self.ignn_cells:
            nodes, edges= layer(nodes, edges, graph, edge_weights)
        
        new_embeddings = self.output_layer(nodes)
        new_embeddings = nn.functional.normalize(new_embeddings) 
        
        return new_embeddings, nodes
    
class HierarchicalGNNBlock(ObjectCondensationBase):

    """
    An interaction network class
    """

    def __init__(self, hparams, num_edges, num_super_edges, num_bipartite_edges, num_iterations, input_size, hidden_size):
        super().__init__(hparams)
            
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """                
            
        # Setup input network
        self.node_encoder = make_mlp(
            hparams["spatial_channels"] + input_size,
            hidden_size,
            hidden_size - hparams["emb_dim"],
            hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["spatial_channels"]+hparams["emb_dim"]),
            hidden_size,
            hidden_size,
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.supernode_encoder = make_mlp(
            hparams["emb_dim"] + input_size,
            hidden_size,
            hidden_size - hparams["emb_dim"],
            hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.superedge_encoder = make_mlp(
            2 * hidden_size,
            hidden_size,
            hidden_size,
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # Initialize GNN blocks
        if hparams["share_weight"]:
            cell = HierarchicalGNNCell(hparams, hidden_size)
            hgnn_cells = [
                cell
                for _ in range(num_iterations)
            ]
        else:
            hgnn_cells = [
                HierarchicalGNNCell(hparams, hidden_size)
                for _ in range(num_iterations)
            ]
        
        self.hgnn_cells = nn.ModuleList(hgnn_cells)
        
        # output layers
        self.output_layer = make_mlp(
            hidden_size,
            hidden_size,
            hparams["emb_dim"],
            hparams["output_layers"],
            layer_norm=hparams["layernorm"],
            output_activation= None,
            hidden_activation=hparams["hidden_output_activation"],
        )
        
        self.hdbscan_model = HDBSCAN(min_cluster_size = hparams["min_cluster_size"])
        self.graph_construction = DynamicGraphConstruction("sigmoid", hparams)
        self.super_graph_construction = DynamicGraphConstruction("sigmoid", hparams)
        self.bipartite_graph_construction = DynamicGraphConstruction("exp", hparams)
        self.num_edges = num_edges
        self.num_super_edges = num_super_edges
        self.num_bipartite_edges = num_bipartite_edges

        
    def forward(self, x, embeddings, old_nodes):
        
        x.requires_grad = True
        
        with torch.no_grad():
            
            clustering_input = embeddings + torch.normal(0, 2e-3, embeddings.shape, device = embeddings.device)
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
        
        graph, edge_weights = self.graph_construction(embeddings, embeddings, sym = True, norm = False, k = self.num_edges)
        super_graph, super_edge_weights = self.super_graph_construction(means, means, sym = True, norm = False, k = self.num_super_edges)
        bipartite_graph, bipartite_edge_weights = self.bipartite_graph_construction(embeddings, means, sym = False, norm = True, k = self.num_bipartite_edges)
        
        del clusters
        
        nodes = torch.cat([checkpoint(self.node_encoder, torch.cat([x, old_nodes], dim = -1)), embeddings], dim = -1)
        edges = checkpoint(self.edge_encoder, torch.cat([x[graph[0]], embeddings[graph[0]], x[graph[1]], embeddings[graph[1]]], dim=1))
        
        supernodes = scatter_add((old_nodes[bipartite_graph[0]])*bipartite_edge_weights, bipartite_graph[1], dim=0, dim_size=means.shape[0])
        supernodes = torch.cat([supernodes, means], dim = -1)
        supernodes = torch.cat([means, checkpoint(self.supernode_encoder, supernodes)], dim = -1)
        superedges = checkpoint(self.superedge_encoder, torch.cat([supernodes[super_graph[0]], supernodes[super_graph[1]]], dim=1))
        
        del means, old_nodes, embeddings
        
        for layer in self.hgnn_cells:
            nodes, edges, supernodes, superedges = layer(nodes,
                                                         edges,
                                                         supernodes,
                                                         superedges,
                                                         graph,
                                                         edge_weights,
                                                         bipartite_graph,
                                                         bipartite_edge_weights,
                                                         super_graph,
                                                         super_edge_weights)
        
        return nodes, supernodes, bipartite_graph
    
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
            
    
        self.mlp_encoder = make_mlp(
            hparams["spatial_channels"] + hparams["cell_channels"],
            hparams["mlp_hidden"],
            hparams["emb_dim"],
            hparams["nb_mlp_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation="Tanh",
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

        # Initialize GNN blocks
        ignn_blocks = [InteractionGNNBlock(hparams, edges, iters, size) for edges, iters, size in zip(self.hparams["pyramid_interaction_edges"], self.hparams["pyramid_interaction_iterations"], self.hparams["pyramid_interaction_sizes"])]
        
        self.ignn_blocks = nn.ModuleList(ignn_blocks)
        self.hgnn_block = HierarchicalGNNBlock(hparams,
                                               self.hparams["pyramid_hierarchical_edges"],
                                               self.hparams["supergraph_sparsity"],
                                               self.hparams["bipartitegraph_sparsity"],
                                               self.hparams["n_hierarchical_graph_iters"],
                                               self.hparams["pyramid_interaction_sizes"][-1],
                                               self.hparams["latent"])
        
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
        embeddings = checkpoint(self.mlp_encoder, torch.cat([x, cell_info], dim = -1))
        
        for block in self.ignn_blocks:
            embeddings, nodes = block(x, embeddings)
        
        nodes, supernodes, bipartite_graph = self.hgnn_block(x, embeddings, nodes)       
            
        bipartite_edge_scores = checkpoint(self.bipartite_edge_scoring,
                                           torch.cat([nodes[bipartite_graph[0]], supernodes[bipartite_graph[1]]], dim=-1)).squeeze()
        bipartite_edge_scores = torch.sigmoid(bipartite_edge_scores)
        supernodes = checkpoint(self.output_layer, supernodes)
        
        return embeddings, bipartite_edge_scores, bipartite_graph, supernodes.squeeze()
