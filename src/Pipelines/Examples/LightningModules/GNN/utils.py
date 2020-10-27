import os, sys

import torch.nn as nn
import torch
import numpy as np
import cupy as cp

def load_dataset(input_dir, num):
    if input_dir is not None:
        all_events = os.listdir(input_dir)
        all_events = sorted([os.path.join(input_dir, event) for event in all_events])
        loaded_events = [torch.load(event, map_location=torch.device('cpu')) for event in all_events[:num]]
        return loaded_events
    else:
        return None

def random_edge_slice_v2(delta_phi, batch):
    '''
    Same behaviour as v1, but avoids the expensive calls to np.isin and np.unique, using sparse operations on GPU
    '''
    # 1. Select random phi
    random_phi = np.random.rand()*2 - 1
    e = batch.edge_index.to('cpu').numpy()
    x = batch.x.to('cpu')

    # 2. Find edges within delta_phi of random_phi
    e_average = (x[e[0], 1] + x[e[1], 1])/2
    dif = abs(e_average - random_phi)
    subset_edges = ((dif < delta_phi) | ((2-dif) < delta_phi)).numpy()

    # 3. Find connected edges to this subset   
    e_ones = cp.array([1]*e_length).astype('Float32')
    subset_ones = cp.array([1]*subset_edges.sum()).astype('Float32')

    e_csr_in = cp.sparse.coo_matrix((e_ones, (cp.array(e[0]).astype('Float32'), cp.arange(e_length).astype('Float32'))), shape=(e.max()+1,e_length)).tocsr()
    e_csr_out = cp.sparse.coo_matrix((e_ones, (cp.array(e[0]).astype('Float32'), cp.arange(e_length).astype('Float32'))), shape=(e.max()+1,e_length)).tocsr()
    e_csr = e_csr_in + e_csr_out

    subset_csr_in = cp.sparse.coo_matrix((subset_ones, (cp.array(e[0, subset_edges]).astype('Float32'), cp.arange(e_length)[subset_edges].astype('Float32'))), shape=(e.max()+1,e_length)).tocsr()
    subset_csr_out = cp.sparse.coo_matrix((subset_ones, (cp.array(e[0, subset_edges]).astype('Float32'), cp.arange(e_length)[subset_edges].astype('Float32'))), shape=(e.max()+1,e_length)).tocsr()
    subset_csr = subset_csr_in + subset_csr_out

    summed = (subset_csr.T * e_csr).sum(axis=0)
    subset_edges_extended = (summed>0)[0].get()
    
    return subset_edges, subset_edges_extended

def random_edge_slice(delta_phi, batch):
    # 1. Select random phi
    random_phi = np.random.rand()*2 - 1
    e = batch.edge_index.to('cpu')
    x = batch.x.to('cpu')

    # 2. Find hits within delta_phi of random_phi
    dif = abs(x[:,1] - random_phi)
    subset_hits = np.where((dif < delta_phi) | ((2-dif) < delta_phi))[0]

    # 3. Filter edges with subset_hits
    subset_edges_ind = (np.isin(e[0], subset_hits) | np.isin(e[1], subset_hits))

    subset_hits = np.unique(e[:, subset_edges_ind])
    subset_edges_extended = (np.isin(e[0], subset_hits) | np.isin(e[1], subset_hits))
    nested_ind = np.isin(np.where(subset_edges_extended)[0], np.where(subset_edges_ind)[0])
    
    return subset_edges_ind, subset_edges_extended, nested_ind

def make_mlp(input_size, sizes,
             hidden_activation='ReLU',
             output_activation='ReLU',
             layer_norm=False):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i+1]))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)