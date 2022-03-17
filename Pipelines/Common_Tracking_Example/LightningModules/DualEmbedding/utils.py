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

def load_dataset_paths(input_dir, num):
    
    all_events = os.listdir(input_dir)
    all_events = sorted([os.path.join(input_dir, event) for event in all_events])[:num]
    
    return all_events

class DualEmbeddingDataset(Dataset):
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
        if self.hparams["use_bidir_truth"]:
            event.true_edges = torch.cat([event.modulewise_true_edges,
                                          event.modulewise_true_edges.flip(0)], dim = 1)
        else:
            event.true_edges = event.modulewise_true_edges
        

        return event
    
    def __len__(self):
        return self.num
    
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