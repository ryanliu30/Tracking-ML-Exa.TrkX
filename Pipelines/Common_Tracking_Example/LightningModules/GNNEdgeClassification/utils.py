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
        
        event.graph = event.edge_index
        event.y = event.y.bool()
        event.y_pid = event.y_pid.bool()
        event.pid_signal = event.pid_signal.bool()
        event.scores = event.scores
        
        if self.hparams["primary"]:
            event.pid[(event.nhits < 3) | (event.primary != 1)] = 0
        else:
            event.pid[(event.nhits < 3)] = 0
            
            
        if self.hparams["cheat"]:
            pid_mask = (event.pid != 0)
            inverse_mask = torch.zeros(len(event.pid), device = self.device).long()
            inverse_mask[pid_mask] = torch.arange(pid_mask.sum(), device = self.device)
            
            for i in ["y", "y_pid", "pid_signal", "scores"]:
                graph_mask = (event.pid[event.graph] != 0).all(0)
                event[i] = event[i][graph_mask]
                
            for i in ["modulewise_true_edges", "signal_true_edges", "edge_index", "graph"]:
                event[i] = event[i][:,(event.pid[event[i]] != 0).all(0)]
                event[i] = inverse_mask[event[i]]
                
            for i in ["x", "cell_data", "pid", "hid", "pt", "primary", "nhits"]:
                event[i] = event[i][pid_mask]
                
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
    
def efficiency_performance_wrt_distance(gnn_graph, pred_graph, truth_graph, n_hop):
    
    array_size = max(gnn_graph.max().item(), pred_graph.max().item(), truth_graph.max().item()) + 1

    if torch.is_tensor(pred_graph):
        l1 = pred_graph.cpu().numpy()
    else:
        l1 = pred_graph
    if torch.is_tensor(truth_graph):
        l2 = truth_graph.cpu().numpy()
    else:
        l2 = truth_graph
    if torch.is_tensor(gnn_graph):
        l3 = gnn_graph.cpu().numpy()
    else:
        l3 = gnn_graph
        
    e_pred = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()
    e_truth = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()
    e_gnn = sp.sparse.coo_matrix(
        (np.ones(l3.shape[1]), l3), shape=(array_size, array_size)
    ).tocsr()
    
    # symmetrization:
    e_pred = (e_pred + e_pred.T) > 0
    e_truth = (e_truth + e_truth.T) > 0
    e_gnn = (e_gnn + e_gnn.T) > 0
    
    # find n hop neighbors
    
    n_hop_neighbors = []
    
    for i in range(n_hop):
        power = e_gnn
        for j in range(i):
            power = power * e_gnn
        n_hop_neighbors.append(power > 0)
        del power
    
    for i in range(n_hop-1, -1, -1):
        for j in range(i-1, -1, -1):
            n_hop_neighbors[i] = n_hop_neighbors[i] - n_hop_neighbors[j]
        n_hop_neighbors[i] = n_hop_neighbors[i] > 0
    
    n_hop_eff = []
    
    for i in range(n_hop):
        signal_num = e_truth.multiply(n_hop_neighbors[i]).sum()
        found_num = e_truth.multiply(e_pred.multiply(n_hop_neighbors[i])).sum()
        n_hop_eff.append("eff:{:.3f} with {}/{}".format(found_num/(signal_num + 1e-12), found_num, signal_num))
        
    return n_hop_eff
    
def make_mlp(
    input_size,
    hidden_size,
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

def find_neighbors(embedding1, embedding2, r_max=1.0, k_max=10):
    embedding1 = embedding1.reshape((1, embedding1.shape[0], embedding1.shape[1]))
    embedding2 = embedding2.reshape((1, embedding2.shape[0], embedding2.shape[1]))
    
    _, idxs, _, _ = frnn.frnn_grid_points(points1 = embedding1,
                                          points2 = embedding2,
                                          lengths1 = None,
                                          lengths2 = None,
                                          K = k_max,
                                          r = r_max,
                                         )
    return idxs.squeeze(0)

def build_graph(embeddings, k, r):
    
    idxs = find_neighbors(embeddings.clone().detach(), embeddings.clone().detach(), r_max=r, k_max=k)

    positive_idxs = idxs >= 0
    ind = torch.arange(idxs.shape[0], device = positive_idxs.device).unsqueeze(1).expand(idxs.shape)
    edges = torch.stack([ind[positive_idxs],
                        idxs[positive_idxs]
                        ], dim = 0)     
    return edges