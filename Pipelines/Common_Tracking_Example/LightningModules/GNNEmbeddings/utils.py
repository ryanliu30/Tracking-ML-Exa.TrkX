import sys
import os
import logging

import torch
import scipy as sp
import numpy as np
from scipy import sparse
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch_geometric.data import Data
import frnn
from cuml.cluster import HDBSCAN
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

import functools
import warnings
from typing import Type

def ignore_warning(warning: Type[Warning]):
    """
    Ignore a given warning occurring during method execution.

    Args:
        warning (Warning): warning type to ignore.

    Returns:
        the inner function

    """

    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category= warning)
                return func(*args, **kwargs)

        return wrapper

    return inner

def load_dataset_paths(input_dir, datatype_names):
    all_events = []
    for name in datatype_names:
        events = os.listdir(os.path.join(input_dir, name))
        events = sorted([os.path.join(input_dir, name, event) for event in events])
        all_events.extend(events)
    # random.shuffle(all_events)
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
        
        if self.hparams["use_toy"]:
            return generate_toys(self.hparams["num_tracks"],
                                 self.hparams["track_dis_width"],
                                 self.hparams["num_layers"],
                                 self.hparams["min_r"],
                                 self.hparams["max_r"],
                                 self.hparams["detector_width"],
                                 self.hparams["ptcut"],
                                 self.hparams["toy_eff"],
                                 self.hparams["toy_pur"]
                                )
        
        event = torch.load(self.dirs[key], map_location=torch.device(self.device))
        if self.hparams["noise"]:
            mask = torch.tensor(len(event.pid) * [False], device = self.device)
            conncted_nodes = torch.unique(event.edge_index[:, event.scores > self.hparams["score_cut"]])
            mask[conncted_nodes] = True
        else:
            mask = (event.pid != 0)
        if self.hparams["hard_ptcut"]:
            mask = mask & (event.pt > self.hparams["ptcut"])

        inverse_mask = torch.zeros(len(event.pid)).long()
        inverse_mask[mask] = torch.arange(mask.sum())
        
        event.graph = event.edge_index
        
        if self.hparams["primary"]:
            event.pid[(event.nhits <= 3) | (event.primary != 1)] = 0
        else:
            event.pid[(event.nhits <= 3)] = 0

        for i in ["y", "y_pid", "pid_signal", "scores"]:
            graph_mask = mask[event.graph].all(0)
            event[i] = event[i][graph_mask]

        for i in ["modulewise_true_edges", "signal_true_edges", "edge_index", "graph"]:
            event[i] = event[i][:, mask[event[i]].all(0)]
            event[i] = inverse_mask[event[i]]

        for i in ["x", "cell_data", "pid", "hid", "pt", "primary", "nhits"]:
            event[i] = event[i][mask]
            
        return event
    
    def __len__(self):
        return self.num

###
   
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

@ignore_warning(RuntimeWarning)
def generate_toys(num_tracks, track_dis_width, num_layers, min_r, max_r, detector_width, ptcut, eff, pur):
    # pT is defined as 1000r
    
    tracks = []
    num_tracks = random.randint(num_tracks-track_dis_width, num_tracks+track_dis_width)
    for i in range(num_tracks):
        r = np.random.uniform(min_r, max_r)
        theta = np.random.uniform(0, np.pi)
        sign = np.random.choice([-1, 1])

        x = np.linspace(0.05, detector_width + 0.05, num = num_layers)
        y = sign*(np.sqrt(r**2 - (x - r*np.cos(theta))**2) - r*np.sin(theta))
        pid = np.array(len(x)*[i+1], dtype = np.int64)
        pt = 1000 * np.array(len(x)*[r])
        
        mask = (y == y)
        x, y, pid, pt = x[mask], y[mask], pid[mask], pt[mask]
        
        tracks.append(np.vstack([x, y, pid, pt]).T)
    
    node_feature = np.concatenate(tracks, axis = 0)
    
    connections = (node_feature[:-1, 2] == node_feature[1:,2])

    idxs = np.arange(len(node_feature))

    truth_graph = np.vstack([idxs[:-1][connections], idxs[1:][connections]])
    signal_true_graph = truth_graph[:, (node_feature[:, 3][truth_graph] > ptcut).all(0)]
    
    fully_connected_graph = np.vstack([np.resize(idxs, (len(idxs),len(idxs))).flatten(), np.resize(idxs, (len(idxs),len(idxs))).T.flatten()])
    del_x = (node_feature[fully_connected_graph[1], 0] - node_feature[fully_connected_graph[0], 0])
    del_y = np.abs((node_feature[fully_connected_graph[0], 1] - node_feature[fully_connected_graph[1], 1]))
    fully_connected_graph = fully_connected_graph[:, (del_x <= 2*detector_width/num_layers) & (del_x > 0) & (del_y<0.4)]
    fully_connected_graph = fully_connected_graph[:, fully_connected_graph[0] != fully_connected_graph[1]]

    truth_graph_samples = signal_true_graph[:, np.random.random_sample(size=signal_true_graph.shape[1]) < eff]
    fake_graph_samples = fully_connected_graph[:, np.random.random_sample(size=fully_connected_graph.shape[1]) < (1-pur)/pur*truth_graph_samples.shape[1]/fully_connected_graph.shape[1]]

    graph = np.concatenate([truth_graph_samples, fake_graph_samples], axis = 1)
    
    graph, y = graph_intersection(graph, signal_true_graph)
    
    y_pid = (node_feature[:,2][graph[0]] == node_feature[:,2][graph[1]])
    pid_signal = (node_feature[:,2][graph[0]] == node_feature[:,2][graph[1]]) & (node_feature[:,3][graph]).all(0)
    node_feature = torch.tensor(node_feature).float()
    
    event = Data(x=node_feature[:,0:2],
                 edge_index= graph,
                 graph = graph,
                 modulewise_true_edges = torch.tensor(truth_graph),
                 signal_true_edges = torch.tensor(signal_true_graph),
                 y=y,
                 pt = node_feature[:,3],
                 pid = node_feature[:,2].long(),
                 y_pid = y_pid,
                 pid_signal = pid_signal,
                )
    _, inverse, counts = event.pid.unique(return_inverse = True, return_counts = True)
    event.nhits = counts[inverse]
    event.pid[(event.nhits <= 3)] = 0
    
    event.signal_true_edges = event.signal_true_edges[:, (event.nhits[event.signal_true_edges] > 3).all(0)]
    
    return event
    
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
    
def find_neighbors(embedding1, embedding2, r_max=1.0, k_max=10):
    embedding1 = embedding1.clone().detach().reshape((1, embedding1.shape[0], embedding1.shape[1]))
    embedding2 = embedding2.clone().detach().reshape((1, embedding2.shape[0], embedding2.shape[1]))
    
    dists, idxs, _, _ = frnn.frnn_grid_points(points1 = embedding1,
                                          points2 = embedding2,
                                          lengths1 = None,
                                          lengths2 = None,
                                          K = k_max,
                                          r = r_max,
                                         )
    return idxs.squeeze(0)

def FRNN_graph(embeddings, r, k):
    
    idxs = find_neighbors(embeddings, embeddings, r_max=r, k_max=k)

    positive_idxs = (idxs.squeeze() >= 0)
    ind = torch.arange(idxs.shape[0], device = positive_idxs.device).unsqueeze(1).expand(idxs.shape)
    edges = torch.stack([ind[positive_idxs],
                        idxs[positive_idxs]
                        ], dim = 0)     
    return edges
        
        
        
    