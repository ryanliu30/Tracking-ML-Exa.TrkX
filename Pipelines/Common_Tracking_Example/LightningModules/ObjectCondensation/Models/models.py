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
        
    def forward(self, nodes, edges, graph):
        
        # Compute new node features
        edge_messages = scatter_add(edges, graph[1], dim=0, dim_size=nodes.shape[0])
        node_input = torch.cat([nodes, edge_messages], dim=-1)
            
        nodes = checkpoint(self.node_network, node_input) + nodes
            
        # Compute new edge features
        edge_input = torch.cat([nodes[graph[0]], nodes[graph[1]], edges], dim=-1)
        edges = checkpoint(self.edge_network, edge_input) + edges
        
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
        
        self.bipartite_edge_scoring = make_mlp(
            2*hparams["latent"],
            hparams["hidden"],
            1,
            hparams["output_layers"],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation="Tanh",
        )
        
        self.softmax = nn.Softmax(dim = -1)
        
        self.hparams = hparams
        
    def forward(self, nodes, edges, supernodes, superedges, graph, bipartite_graph, bipartite_graph_attention_logits, super_graph, super_graph_attention):
        
        # Compute new bipartite graph attention
        bipartite_graph_attention = torch.exp(bipartite_graph_attention_logits + checkpoint(self.bipartite_edge_scoring, torch.cat([nodes[bipartite_graph[0]], supernodes[bipartite_graph[1]]], dim=-1)).squeeze())
        bipartite_graph_attention = bipartite_graph_attention/(1e-12 + \
                                    scatter_add(bipartite_graph_attention, bipartite_graph[0], dim=0, dim_size = nodes.shape[0])[bipartite_graph[0]])
        bipartite_graph_attention = bipartite_graph_attention.unsqueeze(1)
        
        # Compute new supernode features
        node_messages = scatter_add(bipartite_graph_attention*nodes[bipartite_graph[0]], bipartite_graph[1], dim=0, dim_size=bipartite_graph[1].max()+1)
        attention_messages = scatter_add(superedges[super_graph[0]]*super_graph_attention, super_graph[1], dim=0, dim_size=supernodes.shape[0])
        supernodes = checkpoint(self.supernode_network, torch.cat([supernodes, attention_messages, node_messages], dim=-1)) + supernodes
        
        # Compute original graph updates
        supernode_messages = scatter_add(bipartite_graph_attention*supernodes[bipartite_graph[1]], bipartite_graph[0], dim=0, dim_size=nodes.shape[0])
        edge_messages = scatter_add(edges, graph[1], dim=0, dim_size=nodes.shape[0])
        nodes = checkpoint(self.node_network, torch.cat([nodes, edge_messages, supernode_messages], dim=-1)) + nodes
        
        # Compute new superedge features
        superedges = checkpoint(self.superedge_network, torch.cat([supernodes[super_graph[0]], supernodes[super_graph[1]], superedges], dim=-1))\
                    + superedges
        
        # Compute new edge features
        edges = checkpoint(self.edge_network, torch.cat([nodes[graph[0]], nodes[graph[1]], edges], dim=-1)) + edges
        
        return nodes, edges, supernodes, superedges
    
class HierarchicalGNNAttention(nn.Module):
    def __init__(self, hparams, hdbscanmodel, logging):
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
        
        self.bipartite_normalization = nn.BatchNorm1d(1)       
        
        if hparams["clustering_model"] == "DBSCAN":
            self.model = DBSCAN(min_samples = hparams["min_samples"], eps = hparams["DBSCAN_eps"]) 
        if hparams["clustering_model"] == "HDBSCAN":
            self.model = HDBSCAN(min_cluster_size = hparams["min_cluster_size"])
        if hparams["clustering_model"] == "KMEANS":
            self.model = KMeans(n_clusters = hparams["kmeans"])
        
        self.log = logging
        
        self.hparams = hparams
        
    def get_clusters(self, nodes):
        
        # Compute Embeddings
        embeddings = self.clustering_network(nodes)
        embeddings = nn.functional.normalize(embeddings) + torch.normal(0, 2e-3, embeddings.shape, device = embeddings.device)
       
        # Clustering
        clustering_input = cudf.DataFrame(embeddings)        
        clusters = self.model.fit_predict(clustering_input)

        del clustering_input
        
        clusters = torch.as_tensor(clusters, device = nodes.device).long()
        if (clusters >= 0).any():
            clusters[clusters >= 0] = clusters[clusters >= 0].unique(return_inverse = True)[1]
        if (clusters < 0).all():
            clusters = clusters + 1
        
        return clusters
        
    def forward(self, embedding_nodes, encoded_nodes, clusters):
        
        self.device = embedding_nodes.device
        
        embeddings = self.clustering_network(embedding_nodes)
        embeddings  = nn.functional.normalize(embeddings)
    
        self.log("clusters", float(clusters.max().item()+1))

        # Compute Centers
        means = scatter_mean(embeddings[clusters>=0], clusters[clusters>=0], dim=0, dim_size=clusters.max()+1)
        means = nn.functional.normalize(means)
        
        # Construct Bipartite Graph
        bipartite_idxs = find_neighbors(embeddings, means, r_max=1.0, k_max=self.hparams["bipartitegraph_sparsity"])
        positive_idxs = (bipartite_idxs >= 0)
        ind = torch.arange(bipartite_idxs.shape[0], device = self.device).unsqueeze(1).expand(bipartite_idxs.shape)
        bipartite_graph = torch.stack([ind[positive_idxs],
                                       bipartite_idxs[positive_idxs]], dim = 0)
        
        # Compute bipartite attention
        attention = torch.einsum('ij,ij->i', embeddings[bipartite_graph[0]], means[bipartite_graph[1]]) 
        bipartite_graph_attention_logits = self.bipartite_normalization(attention.unsqueeze(1)).squeeze()
        bipartite_graph_attention = torch.exp(bipartite_graph_attention_logits)
        bipartite_graph_attention = bipartite_graph_attention/(1e-12 + \
                                    scatter_add(bipartite_graph_attention, bipartite_graph[0], dim=0, dim_size = embedding_nodes.shape[0])[bipartite_graph[0]])
        bipartite_graph_attention = bipartite_graph_attention.unsqueeze(1)
        
        # Supergraph Construction
        super_idxs = find_neighbors(means, means, r_max=1.0, k_max=self.hparams["supergraph_sparsity"])
        positive_idxs = (super_idxs >= 0)
        ind = torch.arange(super_idxs.shape[0], device = self.device).unsqueeze(1).expand(super_idxs.shape)
        src, dst = symmetrize(cudf.Series(ind[positive_idxs]), cudf.Series(super_idxs[positive_idxs]))       
        super_graph = torch.tensor(cp.vstack([src.to_cupy(), dst.to_cupy()]), device=self.device).long()
        
        # Supergraph Attention
        attention = torch.einsum("ij, ij -> i", means[super_graph[0]], means[super_graph[1]])
        att_max = scatter_max(attention, super_graph[1], dim=0, dim_size = means.shape[0])[0][super_graph[1]]
        att_min = scatter_min(attention, super_graph[1], dim=0, dim_size = means.shape[0])[0][super_graph[1]]
        attention = 2*(attention - att_min)/(1e-12 + (att_max - att_min).detach())
        attention = torch.tanh(attention)
        super_graph_attention = attention.unsqueeze(1)
        
        # Aggregate Supernode Features
        node_messages = checkpoint(self.supernode_encoder, encoded_nodes) 
        supernodes = scatter_add((node_messages[bipartite_graph[0]])*bipartite_graph_attention, bipartite_graph[1], dim=0, dim_size=bipartite_graph[1].max()+1)
        superedges = checkpoint(self.superedge_encoder, torch.cat([supernodes[super_graph[0]], supernodes[super_graph[1]]], dim=1))
        
        return embeddings, supernodes, superedges, bipartite_graph, bipartite_graph_attention_logits, super_graph, super_graph_attention

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
            checkpoint = lambda i, j: i(j)
            
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
        
        # Setup the second input network
        self.second_node_encoder = make_mlp(
            in_channels,
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
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
        
        self.supergraph_construction = HierarchicalGNNAttention(hparams, self.HDBSCANmodel, self.log)

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

        
    def forward(self, x, graph):
        
        x.requires_grad = True
        
        directed_graph = torch.cat([graph, graph.flip(0)], dim = 1)            
        
        nodes = checkpoint(self.node_encoder, x)
        edges = checkpoint(self.edge_encoder, torch.cat([x[directed_graph[0]], x[directed_graph[1]]], dim=1))
        
        for layer in self.ignn_blocks:
            nodes, edges= layer(nodes, edges, directed_graph)
            
        with torch.no_grad():
            clusters = self.supergraph_construction.get_clusters(nodes)
            
        intermediate_embeddings, supernodes, superedges, bipartite_graph, bipartite_graph_attention_logits, super_graph, super_graph_attention = \
                    self.supergraph_construction(nodes, nodes, clusters)
        
        if self.hparams["update_nodes"]:
            nodes = checkpoint(self.second_node_encoder, x)
        
        for layer in self.hgnn_blocks:
            nodes, edges, supernodes, superedges = layer(nodes,
                                                         edges,
                                                         supernodes,
                                                         superedges,
                                                         directed_graph,
                                                         bipartite_graph,
                                                         bipartite_graph_attention_logits,
                                                         super_graph,
                                                         super_graph_attention)
            
        bipartite_edge_scores = checkpoint(self.bipartite_edge_scoring,
                                           torch.cat([nodes[bipartite_graph[0]], supernodes[bipartite_graph[1]]], dim=-1)).squeeze()
        bipartite_edge_scores = torch.sigmoid(bipartite_edge_scores + bipartite_graph_attention_logits)
        supernodes = supernodes.clone().detach()
        supernodes.requires_grad = True
        supernodes = checkpoint(self.output_layer, supernodes)
        
        return intermediate_embeddings, bipartite_edge_scores, bipartite_graph, 10*supernodes.squeeze()
    
class DualHierarchicalGNN(ObjectCondensationBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        
        if hparams["checkpointing"]:
            from torch.utils.checkpoint import checkpoint
        else:
            global checkpoint
            checkpoint = lambda i, j: i(j)
            
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
        
        # Setup the second input network
        self.second_node_encoder = make_mlp(
            in_channels,
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )
        
        # The edge network computes new edge features from connected nodes
        self.second_edge_encoder = make_mlp(
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
        
        self.supergraph_construction = HierarchicalGNNAttention(hparams, self.HDBSCANmodel, self.log)

        # Initialize GNN blocks
        if hparams["share_weight"]:
            block = InteractionGNNBlock(hparams)
            ignn_embedding_blocks = [
                block
                for _ in range(self.hparams["n_interaction_embedding_graph_iters"])
            ]
        else:
            ignn_embedding_blocks = [
                InteractionGNNBlock(hparams)
                for _ in range(self.hparams["n_interaction_embedding_graph_iters"])
            ]
        
        if hparams["share_weight"]:
            block = InteractionGNNBlock(hparams)
            ignn_encoding_blocks = [
                block
                for _ in range(self.hparams["n_interaction_encoding_graph_iters"])
            ]
        else:
            ignn_encoding_blocks = [
                InteractionGNNBlock(hparams)
                for _ in range(self.hparams["n_interaction_encoding_graph_iters"])
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
        
        self.ignn_embedding_blocks = nn.ModuleList(ignn_embedding_blocks)
        self.ignn_encoding_blocks = nn.ModuleList(ignn_encoding_blocks)
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

        
    def forward(self, x, graph):
        
        x.requires_grad = True
        
        directed_graph = torch.cat([graph, graph.flip(0)], dim = 1)            
        
        embedding_nodes = checkpoint(self.node_encoder, x)
        edges = checkpoint(self.edge_encoder, torch.cat([x[directed_graph[0]], x[directed_graph[1]]], dim=1))
        
        for layer in self.ignn_embedding_blocks:
            embedding_nodes, edges= layer(embedding_nodes, edges, directed_graph)
            
        with torch.no_grad():
            clusters = self.supergraph_construction.get_clusters(embedding_nodes)
        
        encoded_nodes = checkpoint(self.second_node_encoder, x)
        edges = checkpoint(self.second_edge_encoder, torch.cat([x[directed_graph[0]], x[directed_graph[1]]], dim=1))
        
        for layer in self.ignn_encoding_blocks:
            encoded_nodes, edges= layer(encoded_nodes, edges, directed_graph)
            
        intermediate_embeddings, supernodes, superedges, bipartite_graph, bipartite_graph_attention_logits, super_graph, super_graph_attention = \
                    self.supergraph_construction(embedding_nodes, encoded_nodes, clusters)
        
        for layer in self.hgnn_blocks:
            encoded_nodes, edges, supernodes, superedges = layer(encoded_nodes,
                                                         edges,
                                                         supernodes,
                                                         superedges,
                                                         directed_graph,
                                                         bipartite_graph,
                                                         bipartite_graph_attention_logits,
                                                         super_graph,
                                                         super_graph_attention)
            
        bipartite_edge_scores = checkpoint(self.bipartite_edge_scoring,
                                           torch.cat([encoded_nodes[bipartite_graph[0]], supernodes[bipartite_graph[1]]], dim=-1)).squeeze()
        bipartite_edge_scores = torch.sigmoid(bipartite_edge_scores + bipartite_graph_attention_logits)
        supernodes = supernodes.clone().detach()
        supernodes.requires_grad = True
        supernodes = checkpoint(self.output_layer, supernodes)
        
        return intermediate_embeddings, bipartite_edge_scores, bipartite_graph, 10*supernodes.squeeze()