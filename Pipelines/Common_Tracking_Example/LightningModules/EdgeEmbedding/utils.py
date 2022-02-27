import sys
import os
import logging

import torch
import scipy as sp
import numpy as np
from scipy import sparse
import frnn
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_neighbors_list(nodes, edges, device = "cuda"):
    
    if torch.is_tensor(edges):
        edges = edges.cpu().numpy()
    
    # Build adjacency matrix in COO form
    neighbors_list = sparse.coo_matrix((np.ones(edges.shape[1]), edges),
                                   shape=(nodes.shape[0], nodes.shape[0])).tocsr()
    # Symmetrize to get bidirectional graph
    neighbors_list = (neighbors_list + neighbors_list.T) > 0
    
    # Get connectivity of each node
    counts = neighbors_list.sum(axis = 1)
    
    max_neighbors = counts.max()
    
    # Calculate how many -1 paddings needed for each row/colunm
    counts = torch.tensor(max_neighbors - counts).to(device)
    counts[counts < 0] = 0
    
    # Append paddings in new rows (N,N)->(N, N+max neighbors), new spaces are filled with paddings needed
    # Represent paddings with indices greater than N
    indices = torch.arange(nodes.shape[0], nodes.shape[0] + max_neighbors).repeat((nodes.shape[0],1)).long().to(device)
    indices[indices >= counts.to(device) + nodes.shape[0]] = -1
    positive_indices = indices >= 0
    idx = torch.arange(nodes.shape[0]).repeat((max_neighbors,1)).T.to(device)
    
    # turn them into COO format
    complement_row = idx[positive_indices].cpu().numpy()
    complement_col = indices[positive_indices].cpu().numpy()
    neighbors_list = neighbors_list.tocoo()
    
    # Concatenate them to get a larger COO matrix
    neighbors_list = sparse.coo_matrix((np.ones(len(neighbors_list.row) + len(complement_row)),
                                        (np.concatenate([neighbors_list.row, complement_row], axis = 0),
                                          np.concatenate([neighbors_list.col, complement_col], axis = 0)
                                          )
                                       ),
                                       shape=(nodes.shape[0], nodes.shape[0] + max_neighbors)
                                      ).tocsr()
    
    # CSR form can automatically be reshaped into adjacency list if padding is done properly
    neighbors_list = neighbors_list.indices.reshape((nodes.shape[0], max_neighbors))
    neighbors_list = torch.tensor(neighbors_list, device = device)
    
    # Turn the indices appended for padding back into -1
    neighbors_list[neighbors_list >= nodes.shape[0]] = -1
    
    return neighbors_list


def smaple_triplets(nodes, edges, n_triplets_per_node, neighbors = None, device = "cuda"):
    
    # This uitility is not used in current version
    
    if neighbors == None:
        neighbors = build_neighbors_list(nodes, edges)
        
    n_triplets_per_node = min(n_triplets_per_node, neighbors.shape[1]//2)
    
    # Randomize list
    neighbors = neighbors[:, torch.randperm(neighbors.shape[1])]
    indices = torch.argsort((neighbors >= 0).int().reshape(neighbors.shape), descending = True)
    indices = (indices + torch.arange(indices.shape[0], device = device).unsqueeze(1)*indices.shape[1]).flatten()
    neighbors = neighbors.flatten()[indices].reshape(neighbors.shape)
    
    # Slicing samples consecutively
    slices = torch.arange(2*n_triplets_per_node, device = device).reshape(-1, 2)
    ind = torch.arange(neighbors.shape[0], device = device)
    triplet_list = torch.cat(
        [torch.stack([ind, neighbors[:, slices[i,0]], neighbors[:, slices[i, 1]]], dim = 0) for i in range(n_triplets_per_node)],
        dim = 1)
    triplet_list = triplet_list[:,(triplet_list != -1).all(0)]
    
    return triplet_list

def load_dataset_paths(input_dir, num):
    
    all_events = os.listdir(input_dir)
    all_events = sorted([os.path.join(input_dir, event) for event in all_events])[:num]
    
    return all_events

class EdgeEmbeddingDataset(Dataset):
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
        event.idxs = event.idxs.long()
        event.idxs[event.idxs_scores < self.hparams["score_cut"]] = -1
        
        # shuffle the order of doublets of each node to aviod training bias
        
        event.idxs = event.idxs[:, torch.randperm(event.idxs.shape[1])]
        
        # sorting only according to their sign to make paddings come after all doublets
        
        indices = torch.argsort((event.idxs >= 0).int().reshape(event.idxs.shape), descending = True)
        indices = (indices + torch.arange(indices.shape[0], device = self.device).unsqueeze(1)*indices.shape[1]).flatten()
        event.idxs = event.idxs.flatten()[indices].reshape(event.idxs.shape)
        event.idxs = event.idxs.long()
        
        if self.hparams["cheat"]:
            
            # If cheating feature is used, construct the adjacency list by concatenating truth graph with assigned number of fake samples. Note that the 
            # proceeding graph construction will automatically symmetrize the graph so the number is approximately twice the number assigned here
            
            idxs = event.idxs[:, :self.hparams["cheating_doublet_per_node"]]
            positive_idxs = (idxs >= 0)
            cheat_edges = torch.stack([torch.arange(event.pid.shape[0], device = self.device).unsqueeze(1).expand(idxs.shape)[positive_idxs], idxs[positive_idxs]]).long()
            event.idxs = build_neighbors_list(event.pid, torch.cat([cheat_edges, event.modulewise_true_edges], dim = 1), device = self.device).long()
        
        if self.stage == "train":
            
            # ind: the node list
            # idxs: the corresponding doublet list
            
            event.ind = torch.randperm(event.idxs.shape[0])[:self.hparams["n_nodes"]]    
            event.idxs = event.idxs[event.ind,:self.hparams["edges_per_nodes"]]
            
        else:
            
            # get the true edges found in embedding & filtering
            positive_idxs = (event.idxs >= 0)
            found_edges = torch.stack([torch.arange(event.pid.shape[0], device = self.device).unsqueeze(1).expand(event.idxs.shape)[positive_idxs], event.idxs[positive_idxs]]).long()
            
            found_edges, y = graph_intersection(found_edges, event.modulewise_true_edges)
            found_edges = found_edges[:,y == 1]
                
            event.found_edges_nocut = found_edges.clone()
            
            # apply cut to the edges for the cut metrics
            event.found_edges = found_edges[:,(event.pt[found_edges] > self.hparams["signal_pt_cut"]).any(0)].clone()
            event.signal_edges = event.modulewise_true_edges[:,(event.pt[event.modulewise_true_edges] > self.hparams["signal_pt_cut"]).any(0)].clone()
            
            # Instead of random, use chunk to divide the graph
            event.ind = torch.chunk(
                torch.arange(event.idxs.shape[0]),
                self.hparams["n_chunks"]
            )
            event.idxs = torch.chunk(event.idxs, self.hparams["n_chunks"])
        
        return event
    
    def __len__(self):
        return self.num
    
def find_neighbors(embedding1, embedding2, neighbors, r_max=1.0, k_max=10):
    
    lengths = (neighbors >= 0).sum(-1)
    lengths_nozero = lengths.clone().detach()
    
    # FRNN will raise error if length 2 is zero
    lengths_nozero[lengths_nozero==0] = 1

    _, idxs, _, _ = frnn.frnn_grid_points(points1 = embedding1,
                                          points2 = embedding2,
                                          lengths1 = lengths,
                                          lengths2 = lengths_nozero,
                                          K = k_max,
                                          r = r_max,
                                         )
    
    return idxs

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