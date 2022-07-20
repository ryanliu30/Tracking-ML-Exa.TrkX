import torch
import cupy as cp
import numpy as np
from torch_scatter import scatter_min, scatter_sum

def eval_metrics(bipartite_graph, event, pt_cut = 1., nhits_cut = 5, majority_cut = 0.5, primary = True):

    _, clusters, counts = bipartite_graph[1].unique(return_inverse = True, return_counts = True)
    bipartite_graph = bipartite_graph[:, counts[clusters] >= (nhits_cut * majority_cut)] 
    bipartite_graph[1] = bipartite_graph[1].unique(return_inverse = True)[1]
    original_pid, pid, nhits = torch.unique(event.pid, return_inverse = True, return_counts = True)
    
    if primary and ("primary" in event):
        primary_mask = (scatter_sum(event.primary, pid) > 0)
        primary_mask = cp.array(primary_mask.cpu().numpy())
        
    pt = scatter_min(event.pt, pid, dim=0, dim_size = pid.max()+1)[0]
    bipartite_graph, original_pid, pid, pt, nhits = cp.asarray(bipartite_graph), cp.asarray(original_pid), cp.asarray(pid), cp.asarray(pt), cp.asarray(nhits)
    
    pid_cluster_mapping = cp.sparse.coo_matrix((cp.ones(bipartite_graph.shape[1]), (pid[bipartite_graph[0]], bipartite_graph[1])), shape=(pid.max().item()+1, bipartite_graph[1].max().item()+1)).tocsr()

    matching = (pid_cluster_mapping >= majority_cut*pid_cluster_mapping.sum(0)) & (pid_cluster_mapping >= majority_cut*nhits.reshape(-1, 1)) 
    row_match, col_match = cp.where(matching)

    matching_mask = ((pid_cluster_mapping[row_match, col_match] > majority_cut*nhits_cut)[0] & (original_pid[row_match] != 0))
    row_match, col_match = row_match[matching_mask], col_match[matching_mask]

    mask = (pt[row_match] > pt_cut) & (nhits[row_match] >= nhits_cut)
    truth_mask = (pt > pt_cut) & (nhits >= nhits_cut)
    selected_hits = (pt[pid] > pt_cut) & (original_pid[pid] != 0) & (nhits[pid] >= nhits_cut)
    
    if primary:
        mask = mask & primary_mask[row_match]
        truth_mask = truth_mask & primary_mask
        selected_hits = selected_hits & primary_mask[pid]

    track_eff = mask.sum()/truth_mask.sum()
    track_pur = (pid_cluster_mapping[row_match, col_match]/pid_cluster_mapping[:, col_match].sum(0)).mean()
    fake_rate = 1 - mask.sum()/(pid_cluster_mapping.shape[1] - (~matching_mask).sum() - (~mask).sum())    
    hit_eff = pid_cluster_mapping[row_match, col_match][mask.reshape(1, -1)].sum()/(selected_hits).sum()

    return {
        "track_eff": track_eff.item(),
        "track_pur": track_pur.item(),
        "fake_rate": fake_rate.item(),
        "hit_eff": hit_eff.item()
    }

def visualizing_embedding_space(event, embeddings, labels, axs, num_tracks_visualized):
    event = event.cpu()
    x, y, z = event.x[:,2], event.x[:, 0]*torch.cos(event.x[:, 1]*np.pi), event.x[:, 0]*torch.sin(event.x[:, 1]*np.pi)