import sys
import os
import logging

import torch
import scipy as sp
import numpy as np
from scipy import sparse
import frnn
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset_paths(input_dir, datatype_names):
    all_events = []
    for name in datatype_names:
        events = os.listdir(os.path.join(input_dir, name))
        events = sorted([os.path.join(input_dir, name, event) for event in events])
        all_events.extend(events)
    return all_events

class EmbeddingDataset(Dataset):
    def __init__(self, dirs, hparams, stage = "train", device = "cpu"):
        super().__init__()
        self.dirs = dirs
        self.num = len(dirs)
        self.device = device
        self.stage = stage
        self.hparams = hparams
        
    def __getitem__(self, key):
        
        # load the event and cut edges according to assigned score cut
        
        event = torch.load(self.dirs[key], map_location=torch.device(self.device))
        
        event.graph = event.edge_index[:, event.scores > self.hparams["score_cut"]]
        event.y = event.y[event.scores > self.hparams["score_cut"]].bool()
        event.y_pid = event.y_pid[event.scores > self.hparams["score_cut"]].bool()

        
        if self.hparams["embedding_regime"] == "edge":
            event.triplet_graph, event.triplet_y, event.triplet_y_pid = build_triplets(event, event.graph, self.device)
            
            event.signal_triplet_true_graph, _, _ = build_triplets(event, event.signal_true_edges, self.device)
        
        return event
    
    def __len__(self):
        return self.num
    

def graph_intersection(
    pred_graph, truth_graph, using_weights=False, weights_bidir=None
):

    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    if torch.is_tensor(pred_graph):
        l1 = pred_graph.cpu().numpy()
    else:
        l1 = pred_graph
    if torch.is_tensor(truth_graph):
        l2 = truth_graph.cpu().numpy()
    else:
        l2 = truth_graph
    e_1 = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()
    e_2 = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()
    del l1

    e_intersection = e_1.multiply(e_2) - ((e_1 - e_2) > 0)
    del e_1
    del e_2

    if using_weights:
        weights_list = weights_bidir.cpu().numpy()
        weights_sparse = sp.sparse.coo_matrix(
            (weights_list, l2), shape=(array_size, array_size)
        ).tocsr()
        del weights_list
        del l2
        new_weights = weights_sparse[e_intersection.astype("bool")]
        del weights_sparse
        new_weights = torch.from_numpy(np.array(new_weights)[0])

    e_intersection = e_intersection.tocoo()
    new_pred_graph = torch.from_numpy(
        np.vstack([e_intersection.row, e_intersection.col])
    ).long()  # .to(device)
    y = torch.from_numpy(e_intersection.data > 0)  # .to(device)
    del e_intersection

    if using_weights:
        return new_pred_graph, y, new_weights
    else:
        return new_pred_graph, y
    
def make_mlp(
    input_size,
    output_size,
    hidden_layers,
    hidden_activation="GELU",
    output_activation="GELU",
    layer_norm=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    hidden_size = max(input_size, output_size)
    sizes = [input_size] + [hidden_size]*(hidden_layers-1) + [output_size]
    # Hidden layers
    for i in range(hidden_layers-1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)


def build_triplets(batch, graph, device):
    
    undir_graph = torch.cat([graph, graph.flip(0)], dim=-1)
    e_max = undir_graph.max().item()
    
    # convert to cupy
    undir_graph = undir_graph.cpu().numpy().astype(np.float32)
    
    # make some utility objects
    num_edges = undir_graph.shape[1]
    e_ones = np.ones(num_edges, dtype = np.float32)
    e_arange = np.arange(num_edges, dtype = np.float32)
    
    # build sparse edge array
    passing_edges_csr_in = sparse.coo_matrix((e_ones, (undir_graph[0], e_arange)), shape=(e_max+1, num_edges)).tocsr()
    passing_edges_csr_out = sparse.coo_matrix((e_ones, (undir_graph[1], e_arange)), shape=(e_max+1, num_edges)).tocsr()
    
    # convert to triplets
    triplet_edges = passing_edges_csr_out.T * passing_edges_csr_in
    triplet_edges = triplet_edges.tocoo()
    
    # convert back to pytorch
    undirected_triplet_edges = torch.as_tensor(np.stack([triplet_edges.row, triplet_edges.col]), device=device)
    
    # convert back to a single-direction edge list
    directed_map = torch.cat([torch.arange(num_edges/2, device=device), torch.arange(num_edges/2, device=device)]).int()
    directed_triplet_edges = directed_map[undirected_triplet_edges.long()].long()
    directed_triplet_edges = directed_triplet_edges[:, directed_triplet_edges[0] != directed_triplet_edges[1]] # Remove self-loops
    directed_triplet_edges = directed_triplet_edges[:, directed_triplet_edges[0] < directed_triplet_edges[1]] # Remove duplicate edges

    return directed_triplet_edges, batch.y[directed_triplet_edges].all(0), batch.y_pid[directed_triplet_edges].all(0)