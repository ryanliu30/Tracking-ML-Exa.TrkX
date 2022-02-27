import os
import logging
    
import torch
from torch.utils.data import random_split
import scipy as sp
import numpy as np
import pandas as pd
import trackml.dataset
from torch_geometric.data import Dataset

"""
Ideally, we would be using FRNN and the GPU. But in the case of a user not having a GPU, or not having FRNN, we import FAISS as the 
nearest neighbor library
"""

import faiss
import faiss.contrib.torch_utils

try:
    import frnn
    using_faiss = False
except ImportError:
    using_faiss = True
    
if torch.cuda.is_available():
    device = "cuda" 
else:
    device = "cpu"
    using_faiss = True
    

class NoisyEmbeddingDataloader(Dataset):
    def __init__(self, dirs, num, true_edges, signal_ptcut):
        super().__init__()
        self.dirs = dirs
        self.num = num
        self.true_edges = true_edges
        self.signal_ptcut = signal_ptcut
        
    def __getitem__(self, key):
        loaded_event = torch.load(self.dirs[key], map_location=torch.device("cpu"))
        loaded_event = noisy_select_data(loaded_event, self.true_edges, self.signal_ptcut)[0]
        return loaded_event
    
    def __len__(self):
        return self.num
    
class FullEmbeddingDataloader(Dataset):
    def __init__(self, dirs, num, true_edges):
        super().__init__()
        self.dirs = dirs
        self.num = num
        self.true_edges = true_edges
        
    def __getitem__(self, key):
        loaded_event = torch.load(self.dirs[key], map_location=torch.device("cpu"))
        loaded_event = full_select_data(loaded_event, self.true_edges)[0]
        return loaded_event
    
    def __len__(self):
        return self.num
    
class MultiEmbeddingDataloader(Dataset):
    def __init__(self, dirs, num, pt_background_cut, pt_signal_cut, nhits, primary_only, true_edges, noise):
        super().__init__()
        self.dirs = dirs
        self.num = num
        self.pt_background_cut = pt_background_cut
        self.pt_signal_cut = pt_signal_cut
        self.nhits_min = nhits
        self.primary_only = primary_only
        self.true_edges = true_edges
        self.noise = noise
        
    def __getitem__(self, key):
        loaded_event = torch.load(self.dirs[key], map_location=torch.device("cpu"))
        loaded_event = select_data(loaded_event, self.pt_background_cut, self.pt_signal_cut, self.nhits_min, self.primary_only, self.true_edges, self.noise)[0]
        return loaded_event
    
    def __len__(self):
        return self.num    

def full_load_dataset(input_dir, num, true_edges):
    if input_dir is not None:
        all_events = os.listdir(input_dir)
        all_events = sorted([os.path.join(input_dir, event) for event in all_events])[:num]
        loaded_events = FullEmbeddingDataloader(all_events, num, true_edges)
        return loaded_events
    else:
        return None
    
def noisy_load_dataset(input_dir, num, true_edges, signal_ptcut):
    if input_dir is not None:
        all_events = os.listdir(input_dir)
        all_events = sorted([os.path.join(input_dir, event) for event in all_events])[:num]
        loaded_events = NoisyEmbeddingDataloader(all_events, num, true_edges, signal_ptcut)
        return loaded_events
    else:
        return None
    
def split_full_datasets(input_dir="", train_split=[100,10,10], seed = 0, true_edges = None, **kwargs):
    """
    Prepare the random Train, Val, Test split, using a seed for reproducibility. Seed should be
    changed across final varied runs, but can be left as default for experimentation.
    """
    torch.manual_seed(seed)
    loaded_events = full_load_dataset(input_dir, sum(train_split), true_edges)
    train_events, val_events, test_events = random_split(loaded_events, train_split)

    return train_events, val_events, test_events
    

def split_datasets(input_dir="", train_split=[100,10,10], signal_ptcut = 0, seed = 0, true_edges = None, **kwargs):
    """
    Prepare the random Train, Val, Test split, using a seed for reproducibility. Seed should be
    changed across final varied runs, but can be left as default for experimentation.
    """
    torch.manual_seed(seed)
    loaded_events = noisy_load_dataset(input_dir, sum(train_split), true_edges, signal_ptcut)
    train_events, val_events, test_events = random_split(loaded_events, train_split)

    return train_events, val_events, test_events


def split_datasets_old(input_dir="", train_split=[100,10,10], pt_background_cut=0, pt_signal_cut=0, nhits=0, primary_only=False, true_edges=None, noise=True, seed=0, **kwargs):
    """
    Prepare the random Train, Val, Test split, using a seed for reproducibility. Seed should be
    changed across final varied runs, but can be left as default for experimentation.
    """
    
    torch.manual_seed(seed)
    loaded_events = load_dataset(input_dir, sum(train_split),  pt_background_cut, pt_signal_cut, nhits, primary_only, true_edges, noise)
    train_events, val_events, test_events = random_split(loaded_events, train_split)

    return train_events, val_events, test_events

def get_edge_subset(edges, mask_where, inverse_mask):
    
    included_edges_mask = np.isin(edges, mask_where).all(0)    
    included_edges = edges[:, included_edges_mask]
    included_edges = inverse_mask[included_edges]
    
    return included_edges, included_edges_mask

def noisy_select_data(events, true_edges, signal_ptcut):
    # Handle event in batched form
    if type(events) is not list:
        events = [events]

    # signal section
    for event in events:

        signal_mask = (event.pt > signal_ptcut) & (event.pid != 0)
        signal_where = torch.where(signal_mask)[0]

        inverse_mask = torch.zeros(signal_where.max()+1).long()
        inverse_mask[signal_where] = torch.arange(len(signal_where))

        event["signal_true_edges"], _ = get_edge_subset(event[true_edges], signal_where, inverse_mask)

        node_features = ["cell_data", "x", "pid", "pt", "nhits"]
        for feature in node_features:
            event["signal_" + feature] = event[feature][signal_mask]
        
    return events

def full_select_data(events, true_edges):
    # Handle event in batched form
    if type(events) is not list:
        events = [events]

    # signal section
    for event in events:

        signal_mask = (event.pid != 0)
        signal_where = torch.where(signal_mask)[0]

        inverse_mask = torch.zeros(signal_where.max()+1).long()
        inverse_mask[signal_where] = torch.arange(len(signal_where))

        event["signal_true_edges"], _ = get_edge_subset(event[true_edges], signal_where, inverse_mask)

        node_features = ["cell_data", "x", "pid", "pt", "nhits"]
        for feature in node_features:
            event["signal_" + feature] = event[feature][signal_mask]
        
    return events


def load_dataset(input_dir, num, pt_background_cut, pt_signal_cut, nhits, primary_only, true_edges, noise):
    if input_dir is not None:
        all_events = os.listdir(input_dir)
        all_events = sorted([os.path.join(input_dir, event) for event in all_events])[:num]
        loaded_events = MultiEmbeddingDataloader(all_events, num, pt_background_cut, pt_signal_cut, nhits, primary_only, true_edges, noise)
        return loaded_events
    else:
        return None


def select_data(events, pt_background_cut, pt_signal_cut, nhits_min, primary_only, true_edges, noise):
    # Handle event in batched form
    if type(events) is not list:
        events = [events]

    # NOTE: Cutting background by pT BY DEFINITION removes noise
    if pt_background_cut > 0 or not noise:
        for event in events:
            
            pt_mask = (event.pt > pt_background_cut) & (event.pid != 0)
            pt_where = torch.where(pt_mask)[0]
            
            inverse_mask = torch.zeros(pt_where.max()+1).long()
            inverse_mask[pt_where] = torch.arange(len(pt_where))
            
            event[true_edges], edge_mask = get_edge_subset(event[true_edges], pt_where, inverse_mask)

            node_features = ["cell_data", "x", "hid", "pid", "pt", "nhits", "primary"]
            for feature in node_features:
                event[feature] = event[feature][pt_mask]
    
    for event in events:
        
        event.signal_true_edges = event[true_edges]
        
        if ("pt" in event.__dict__.keys()) & ("primary" in event.__dict__.keys()) & ("nhits" in event.__dict__.keys()):
            edge_subset = (
                (event.pt[event[true_edges]] > pt_signal_cut).all(0) &
                (event.nhits[event[true_edges]] >= nhits_min).all(0) &
                (event.primary[event[true_edges]].bool().all(0) | (not primary_only))
            )
        
            event.signal_true_edges = event.signal_true_edges[:, edge_subset]
        
    return events


def reset_edge_id(subset, graph):
    subset_ind = np.where(subset)[0]
    filler = -np.ones((graph.max() + 1,))
    filler[subset_ind] = np.arange(len(subset_ind))
    graph = torch.from_numpy(filler[graph]).long()
    exist_edges = (graph[0] >= 0) & (graph[1] >= 0)
    graph = graph[:, exist_edges]

    return graph, exist_edges


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


def build_edges(query, database, indices=None, r_max=1.0, k_max=10, return_indices=False):
    
    dists, idxs, nn, grid = frnn.frnn_grid_points(points1=query.unsqueeze(0), points2=database.unsqueeze(0), lengths1=None, lengths2=None, K=k_max, r=r_max, grid=None, return_nn=False, return_sorted=True)
    
    idxs = idxs.squeeze().int()
    ind = torch.Tensor.repeat(torch.arange(idxs.shape[0], device=device), (idxs.shape[1], 1), 1).T.int()
    positive_idxs = idxs >= 0
    edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()
    
    # Reset indices subset to correct global index 
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]
    
    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    if return_indices:
        return edge_list, dists, idxs, ind
    else:
        return edge_list

def multi_build_edges(queries, databases, indices=None, r_max=1.0, k_max=10, return_indices=False, stage = "train"):
    
    idxs = torch.empty([1, queries[0].shape[0], 0]).cuda()
    dists = torch.empty([1, queries[0].shape[0], 0]).cuda()
    
    for query, database in zip(queries, databases):
        dist, idx, _, _ = frnn.frnn_grid_points(points1=query.unsqueeze(0), points2=database.unsqueeze(0), lengths1=None, lengths2=None, K=k_max, r=r_max, grid=None, return_nn=False, return_sorted=True)
        idxs = torch.cat([idxs,idx], axis = 2)
        dists = torch.cat([dists,dist], axis = 2)
        
    dists = dists.squeeze().float()
    idxs = idxs.squeeze().int()
    if stage == "test":
        idxs, _ = torch.sort(idxs,dim = 1)
    
    ind = torch.Tensor.repeat(torch.arange(idxs.shape[0], device=device), (idxs.shape[1], 1)).T.int()
    positive_idxs = idxs >= 0
    edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]]).long()
    if stage == "test":
        edge_list = torch.unique_consecutive(edge_list, dim = 0)
    
    # Reset indices subset to correct global index 
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]
    
    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    if return_indices:
        return edge_list, dists, idxs, ind
    else:
        return edge_list

def build_knn(spatial, k):

    if device == "cuda":
        res = faiss.StandardGpuResources()
        _, I = faiss.knn_gpu(res, spatial, spatial, k_max)
    elif device == "cpu":
        index = faiss.IndexFlatL2(spatial.shape[1])
        index.add(spatial)
        _, I = index.search(spatial, k_max)

    ind = torch.Tensor.repeat(
        torch.arange(I.shape[0], device=device), (I.shape[1], 1), 1
    ).T
    edge_list = torch.stack([ind, I])

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return edge_list


def get_best_run(run_label, wandb_save_dir):
    for (root_dir, dirs, files) in os.walk(wandb_save_dir + "/wandb"):
        if run_label in dirs:
            run_root = root_dir

    best_run_base = os.path.join(run_root, run_label, "checkpoints")
    best_run = os.listdir(best_run_base)
    best_run_path = os.path.join(best_run_base, best_run[0])

    return best_run_path


# -------------------------- Performance Evaluation -------------------


def embedding_model_evaluation(model, trainer, fom="eff", fixed_value=0.96):

    # Seed solver with one batch, then run on full test dataset
    sol = root(
        evaluate_set_root,
        args=(model, trainer, fixed_value, fom),
        x0=0.9,
        x1=1.2,
        xtol=0.001,
    )
    print("Seed solver complete, radius:", sol.root)

    # Return ( (efficiency, purity), radius_size)
    return evaluate_set_metrics(sol.root, model, trainer), sol.root


def evaluate_set_root(r, model, trainer, goal=0.96, fom="eff"):
    eff, pur = evaluate_set_metrics(r, model, trainer)

    if fom == "eff":
        return eff - goal

    elif fom == "pur":
        return pur - goal


def get_metrics(test_results, model):

    ps = [len(result["truth"]) for result in test_results]
    ts = [result["truth_graph"].shape[1] for result in test_results]
    tps = [result["truth"].sum() for result in test_results]

    efficiencies = [tp / t for (t, tp) in zip(ts, tps)]
    purities = [tp / p for (p, tp) in zip(ps, tps)]

    mean_efficiency = np.mean(efficiencies)
    mean_purity = np.mean(purities)

    return mean_efficiency, mean_purity


def evaluate_set_metrics(r_test, model, trainer):

    model.hparams.r_test = r_test
    test_results = trainer.test(ckpt_path=None)

    mean_efficiency, mean_purity = get_metrics(test_results, model)

    print(mean_purity, mean_efficiency)

    return mean_efficiency, mean_purity
