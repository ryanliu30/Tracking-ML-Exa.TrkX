import torch
import cupy as cp
import numpy as np
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min
from matplotlib import cm
from sklearn.manifold import TSNE
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def eval_metrics(bipartite_graph, event, pt_cut = 1., nhits_cut = 5, majority_cut = 0.5, primary = True):
    
    if majority_cut < 0.5:
        raise UserWarning("A majority cut less than 50% will cause incorrect measuring of tracking performance")
    if majority_cut == 0.5:
        majority_cut = majority_cut + 1e-12
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
    
    matching = (pid_cluster_mapping >= majority_cut*pid_cluster_mapping.sum(0)) & (pid_cluster_mapping >= majority_cut*nhits.reshape(-1, 1)) & (pid_cluster_mapping == pid_cluster_mapping.max(1).todense())
    row_match, col_match = cp.where(matching)
    if row_match.shape[0] == 0:
        return {
            "track_eff": 0,
            "track_pur": 0,
            "hit_eff": 0,
            "hit_pur": 0
        }
    matching_mask = ((pid_cluster_mapping[row_match, col_match] > majority_cut*nhits_cut)[0] & (original_pid[row_match] != 0))
    row_match, col_match = row_match[matching_mask], col_match[matching_mask]
    if row_match.shape[0] == 0:
        return {
            "track_eff": 0,
            "track_pur": 0,
            "hit_eff": 0,
            "hit_pur": 0
        }
    mask = (pt[row_match] > pt_cut) & (nhits[row_match] >= nhits_cut)
    truth_mask = (pt > pt_cut) & (nhits >= nhits_cut)
    selected_hits = (pt[pid] > pt_cut) & (original_pid[pid] != 0) & (nhits[pid] >= nhits_cut)
    
    if primary:
        mask = mask & primary_mask[row_match]
        truth_mask = truth_mask & primary_mask
        selected_hits = selected_hits & primary_mask[pid]

    track_eff = mask.sum()/truth_mask.sum()
    hit_pur = (pid_cluster_mapping[row_match, col_match]/pid_cluster_mapping[:, col_match].sum(0)).mean()
    track_pur = mask.sum()/(pid_cluster_mapping.shape[1] - (~matching_mask).sum() - (~mask).sum())    
    hit_eff = (pid_cluster_mapping[row_match, col_match][mask.reshape(1, -1)]/(nhits[row_match][mask])).mean()

    return {
        "track_eff": track_eff.item(),
        "track_pur": track_pur.item(),
        "hit_eff": hit_eff.item(),
        "hit_pur": hit_pur.item()
    }

@torch.no_grad()
def visualizing_embedding_space(event, embeddings, labels, num_tracks_visualized = 100, regime = "node embedding", **kwargs):
    
    event = event.cpu()
    labels = labels.cpu()
    embeddings = embeddings.cpu()
    fig = plt.figure(figsize = (20, 20), dpi = 300)
    ax = plt.axes(projection='3d')
    
    selection_mask = event.pid.unique(return_inverse = True)[1]
    selection_mask = torch.randperm(selection_mask.max()+1)[selection_mask]
    selection_mask = ((selection_mask < num_tracks_visualized) | (event.pid == 0))
    
    edge_mask = torch.zeros(len(event.pid)).long()
    edge_mask[selection_mask] = torch.arange(selection_mask.sum())
    
    coords = event.x[selection_mask]
    edges = edge_mask[event.modulewise_true_edges]
    
    embeddings = embeddings[selection_mask]
    labels = labels[selection_mask]
    x, y, z = coords[:,2].detach().numpy(), (coords[:, 0]*torch.cos(coords[:, 1]*np.pi)).detach().numpy(), (coords[:, 0]*torch.sin(coords[:, 1]*np.pi)).detach().numpy()
    
    if regime not in ["node embedding", "mean embedding", "random coloring"]:
        raise ValueError('regime must be one of node embedding, mean embedding, or random coloring')
        
    if regime == "node embedding":
        embeddings = torch.tensor(TSNE(n_components = 1, init = "pca").fit_transform(embeddings.cpu().numpy()))
        embeddings = (embeddings-embeddings.min())/(embeddings.max()-embeddings.min())
        node_color = cm.gist_rainbow(embeddings)
        
    if regime == "mean embedding":
        node_color = cm.Greys(np.ones(len(coords)))
        
        means = nn.functional.normalize(scatter_mean(embeddings[labels >= 0], labels[labels >= 0], dim = 0)).cpu().numpy()
        means = torch.tensor(TSNE(n_components = 1, init = "pca").fit_transform(means))
        means = (means-means.min())/(means.max()-means.min())
    
        node_color[labels >= 0] = cm.gist_rainbow(means[labels[labels >= 0]]).squeeze()
        
    if regime == "random coloring":
        node_color = cm.Greys(np.ones(len(coords)))
        labels[labels >= 0] = torch.randperm(labels.max()+1)[labels[labels >= 0]]
        labels = labels.float()/labels.max()
        node_color[labels >= 0] = cm.gist_rainbow(labels[labels >= 0])
        
    for edge in edges.T:
        ax.plot3D(x[edge], y[edge], z[edge], linestyle='-', **kwargs)
    ax.scatter3D(x, y, z, c=node_color)
    
    return ax
    
        
    
        
        
        
        
    