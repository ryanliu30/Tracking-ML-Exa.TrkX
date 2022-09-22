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
import cudf
import cupy as cp
import numpy as np
from sklearn.mixture import GaussianMixture
import cugraph
from scipy.optimize import fsolve

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
        self.GMM_model = GaussianMixture(n_components = 2)
        self.super_graph_construction = DynamicGraphConstruction("sigmoid", hparams)
        self.bipartite_graph_construction = DynamicGraphConstruction("exp", hparams)
        self.score_cut = nn.parameter.Parameter(data=torch.zeros(1), requires_grad=False)
        
        self.log = logging
        self.hparams = hparams
        
    def determine_cut(self, cut0):
        sigmoid = lambda x: 1/(1+np.exp(-x))
        func = lambda x: sigmoid(self.hparams["cluster_granularity"])*self.GMM_model.predict_proba(x.reshape((-1, 1)))[:, self.GMM_model.means_.argmin()] - sigmoid(-self.hparams["cluster_granularity"])*self.GMM_model.predict_proba(x.reshape((-1, 1)))[:, self.GMM_model.means_.argmax()]
        cut = fsolve(func, cut0)
        return cut.item()
        
    def clustering(self, x, embeddings, graph):
        with torch.no_grad():
            
            # Compute Cosine Likelihood
            likelihood = torch.einsum('ij,ij->i', embeddings[graph[0]], embeddings[graph[1]])
            likelihood = torch.atanh(torch.clamp(likelihood, min=-1+1e-7, max=1-1e-7))
            
            # GMM edge cutting
            self.GMM_model.fit(likelihood.unsqueeze(1).cpu().numpy())
            if self.score_cut == 0:
                cut = self.determine_cut(self.GMM_model.means_.mean().item())
                self.score_cut.data = torch.tensor([cut], device = self.score_cut.device)
            cut = self.determine_cut(self.score_cut.data.item())
            if self.training & (cut < self.GMM_model.means_.max().item()) & (cut > self.GMM_model.means_.min().item()):
                self.score_cut.data = 0.95*self.score_cut.data + 0.05*cut
            else:
                cut = self.determine_cut(self.GMM_model.means_.mean().item())
                if self.training & (cut < self.GMM_model.means_.max().item()) & (cut > self.GMM_model.means_.min().item()):
                    self.score_cut.data = 0.95*self.score_cut.data + 0.05*cut
            
            self.log("score_cut", self.score_cut.data.item())
            
            # Connected Components
            mask = likelihood >= self.score_cut.to(likelihood.device)
            try:
                G = cugraph.Graph()
                df = cudf.DataFrame({"src": cp.asarray(graph[0, mask]),
                                     "dst": cp.asarray(graph[1, mask]),
                                    })            
                G.from_cudf_edgelist(df, source = "src", destination = "dst")
                connected_components = cugraph.components.connected_components(G)
            except ValueError:
                G = cugraph.Graph()
                df = cudf.DataFrame({"src": cp.asarray(graph[0]),
                                     "dst": cp.asarray(graph[1]),
                                    })            
                G.from_cudf_edgelist(df, source = "src", destination = "dst")
                connected_components = cugraph.components.connected_components(G)
            
            # Obtain Clustering Results
            clusters = -torch.ones(len(x), device = x.device).long()
            labels = torch.as_tensor(connected_components["labels"], device = x.device)
            vertex = torch.tensor(connected_components["vertex"], device = x.device) 
            _, inverse, counts = labels.unique(return_inverse = True, return_counts = True)
            mask = counts[inverse] >= self.hparams["min_cluster_size"]
            clusters[vertex[mask]] = labels[mask].unique(return_inverse = True)[1].long()
            
            return clusters
        
    def forward(self, x, embeddings, nodes, edges, graph):
        
        x.requires_grad = True
        
        # Compute clustering
        clusters = self.clustering(x, embeddings, graph)

        # Compute Centers
        means = scatter_mean(embeddings[clusters >= 0], clusters[clusters >= 0], dim=0, dim_size=clusters.max()+1)
        means = nn.functional.normalize(means)
        
        super_graph, super_edge_weights = self.super_graph_construction(means, means, sym = True, norm = True, k = self.hparams["supergraph_sparsity"])
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
    
class Embedding_HierarchicalGNN_GMM(EmbeddingBase):

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

