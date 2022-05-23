import sys

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add

from ..gnn_clustering_base import GNNClusteringBase
from ..utils import make_mlp

class InteractionGNNBlock(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        if hparams["use_toy"]:
            hparams["regime"] = []
            hparams["spatial_channels"] = 2
            
        if hparams["checkpointing"]:
            from torch.utils.checkpoint import checkpoint
        else:
            global checkpoint
            checkpoint = lambda i, j: i(j)
        
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
        
        nodes_res = nodes.clone()
        edges_res = edges.clone()
        
        # Compute new node features
        edge_messages = scatter_add(edges, graph[1], dim=0, dim_size=nodes.shape[0])
        node_inputs = torch.cat([nodes, edge_messages], dim=-1)
            
        nodes = checkpoint(self.node_network, node_inputs) + nodes_res
            
        # Compute new edge features
        edge_input = torch.cat([nodes[graph[0]], nodes[graph[1]], edges], dim=-1)
        edges = checkpoint(self.edge_network, edge_input) + edges_res
        
        return nodes, edges


class HierarchicalGNNBlock(nn.Module):
    def __init__(self, assignment_network, hparams):
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
        self.node_network_1 = make_mlp(
            2 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.node_network_2 = make_mlp(
            2 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        
        if hparams["att_on_supergraph"]:
            
            self.supernode_network = make_mlp(
                3 * hparams["latent"],
                hparams["hidden"],
                hparams["latent"],
                hparams["nb_node_layer"],
                layer_norm=hparams["layernorm"],
                output_activation=hparams["hidden_activation"],
                hidden_activation=hparams["hidden_activation"],
            )

            self.multihead_attention = nn.MultiheadAttention(hparams["latent"], 8)
            
        else:
            self.supernode_network = make_mlp(
                2 * hparams["latent"],
                hparams["hidden"],
                hparams["latent"],
                hparams["nb_node_layer"],
                layer_norm=hparams["layernorm"],
                output_activation=hparams["hidden_activation"],
                hidden_activation=hparams["hidden_activation"],
            )
        
        self.assignment_network = assignment_network
        
        self.softmax = nn.Softmax(dim = -1)
        
        self.hparams = hparams
        
    def forward(self, x, nodes, edges, assignment, supernodes, graph):
        
        # Compute new node features
        supernode_messages = torch.matmul(assignment, supernodes)
        
        node_inputs = torch.cat([nodes, supernode_messages], dim=-1)      
        nodes = checkpoint(self.node_network_1, node_inputs) + nodes
        del supernode_messages, node_inputs
        
        # Update assignment matrix
        unnormalized_assignment = checkpoint(self.assignment_network,torch.cat([nodes, x], dim = -1))
        assignment = self.softmax(unnormalized_assignment)
    
        if self.hparams["att_on_supergraph"]:
            # Compute Transformer Attention
            node_messages = torch.matmul(assignment.T, nodes)
            attention_messages = self.multihead_attention(supernodes.unsqueeze(1), supernodes.unsqueeze(1), supernodes.unsqueeze(1))[0].squeeze()

            # Compute new supernode features
            supernode_inputs = torch.cat([supernodes, attention_messages, node_messages], dim=-1)
            supernodes = checkpoint(self.supernode_network, supernode_inputs) + supernodes
            del node_messages, attention_messages, supernode_inputs
            
        else:
            node_messages = torch.matmul(assignment.T, nodes)
            # Compute new supernode features
            supernode_inputs = torch.cat([supernodes, node_messages], dim=-1)
            supernodes = checkpoint(self.supernode_network, supernode_inputs) + supernodes
            del node_messages, supernode_inputs            
            
        # Compute new edge features
        edge_input = torch.cat([nodes[graph[0]], nodes[graph[1]], edges], dim=-1)
        edges = checkpoint(self.edge_network, edge_input) + edges
        del edge_input
        
        # Compute Interaction Network Updates
        edge_messages = scatter_add(edges, graph[1], dim=0, dim_size=nodes.shape[0])
        node_inputs = torch.cat([nodes, edge_messages], dim=-1) 
        nodes = checkpoint(self.node_network_2, node_inputs) + nodes
        del edge_messages, node_inputs
        
        return nodes, edges, assignment, supernodes, unnormalized_assignment

class InteractionGNN(GNNClusteringBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
            
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
        
        self.assignment_network = make_mlp(
            (hparams["latent"] + in_channels),
            hparams["hidden"],
            hparams["clusters"],
            hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["hidden_activation"],
        )
        
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
            block = HierarchicalGNNBlock(self.assignment_network, hparams)
            hgnn_blocks = [
                block
                for _ in range(self.hparams["n_hierarchical_graph_iters"])
            ]
        else:
            hgnn_blocks = [
                HierarchicalGNNBlock(self.assignment_network, hparams)
                for _ in range(self.hparams["n_hierarchical_graph_iters"])
            ]
        
        self.ignn_blocks = nn.ModuleList(ignn_blocks)
        self.hgnn_blocks = nn.ModuleList(hgnn_blocks)
        

    def forward(self, x, assignment_init, graph):
        
        x.requires_grad = True
        
        directed_graph = torch.cat([graph, graph.flip(0)], dim = 1)
        nodes = checkpoint(self.node_encoder, x)
        edges = checkpoint(self.edge_encoder, torch.cat([x[directed_graph[0]], x[directed_graph[1]]], dim=1))
        
        for layer in self.ignn_blocks:
            nodes, edges= layer(nodes, edges, directed_graph)
        
        assignment = checkpoint(self.assignment_network,torch.cat([nodes, x], dim = -1))   
        supernodes = torch.matmul(assignment.T, nodes)

        for layer in self.hgnn_blocks:
            nodes, edges, assignment, supernodes, unnormalized_assignment = layer(x, nodes, edges, assignment, supernodes, directed_graph)
        
        return assignment, unnormalized_assignment



        