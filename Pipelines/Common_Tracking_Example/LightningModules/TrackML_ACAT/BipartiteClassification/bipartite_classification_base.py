# System imports
import sys, os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
import numpy as np
import cupy as cp
import wandb
from torch.utils.data import random_split
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add, scatter_min
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from cuml.cluster import HDBSCAN, KMeans
from torch_geometric.data import Data
from sklearn.mixture import GaussianMixture


device = "cuda" if torch.cuda.is_available() else "cpu"

# Local imports
sys.path.append("../..")
from utils import TrackMLDataset, load_dataset_paths, FRNN_graph, graph_intersection
from tracking_utils import eval_metrics

class BipartiteClassificationBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """
        self.save_hyperparameters(hparams)
        self.GMM_model = GaussianMixture(n_components = 2)
        
    def setup(self, stage):
        
        # For now I only use single input which is the test set of the upstream pipeline
        paths = load_dataset_paths(self.hparams["input_dir"], self.hparams["datatype_names"])
        paths = paths[:sum(self.hparams["train_split"])]
        self.trainset, self.valset, self.testset = random_split(paths, self.hparams["train_split"], generator=torch.Generator().manual_seed(0))
        
    def train_dataloader(self):
        self.trainset = TrackMLDataset(self.trainset, self.hparams, stage = "train", device = "cpu")
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=16, shuffle = True)
        else:
            return None

    def val_dataloader(self):
        self.valset = TrackMLDataset(self.valset, self.hparams, stage = "val", device = "cpu")
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=16)
        else:
            return None

    def test_dataloader(self):
        self.testset = TrackMLDataset(self.testset, self.hparams, stage = "test", device = "cpu")
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=16)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler
    
    def pt_weighting(self, pt):
        
        pt[pt!=pt] = 0

        h = lambda i: torch.heaviside(i, torch.zeros(1).to(pt))
        minimum = lambda i: torch.minimum(i, torch.ones(1).to(pt))
        
        eps = self.hparams["weight_leak"]
        cut = self.hparams["ptcut"] - self.hparams["pt_interval"]
        cap = self.hparams["ptcut"]
        min_weight = self.hparams["weight_min"]
        
        return min_weight + (1-min_weight)*minimum(h(pt-cut)*(pt-cut)/(cap-cut)) + (eps * h(pt-cap) * (pt-cap))
    
    def get_training_samples(self, embeddings, batch):
        
        prediction_graph = FRNN_graph(embeddings, self.hparams["train_r"], self.hparams["knn"])
        
        e_bidir = torch.cat([batch.modulewise_true_edges, batch.modulewise_true_edges.flip(0)], dim = 1)
        e_bidir = e_bidir[:, batch.signal_mask[e_bidir].all(0)]
        
        if self.hparams["true_edges"] == "modulewise_true_edges":
            
            new_graph, y = graph_intersection(prediction_graph, e_bidir)
            new_graph, y = new_graph.to(self.device), y.to(self.device)
            fake_samples = new_graph[:, y==0]
            PID_mask = (batch.pid[fake_samples[0]] != batch.pid[fake_samples[1]]) | (batch.pid[fake_samples] == 0).any(0)
            fake_samples = fake_samples[:, PID_mask]
            new_graph = torch.cat([fake_samples, e_bidir], dim = 1)
            y = torch.cat([torch.zeros(fake_samples.shape[1], device = self.device),
                           torch.ones(e_bidir.shape[1], device = self.device)], dim = 0).bool()
            
        if self.hparams["true_edges"] == "pid_true_edges":
            
            new_graph = torch.cat([prediction_graph, e_bidir], dim = 1)
            y = (batch.pid[new_graph[0]] == batch.pid[new_graph[1]]) & (batch.pid[new_graph] != 0).all(0)
            mask = (batch.signal_mask[new_graph]).all(0) | y == 0
            new_graph = new_graph[:, mask]
            y = y[mask]
            
        return new_graph, y
    
    def get_emb_weight(self, batch, graph, y):
        
        weights = self.pt_weighting(batch.pt[graph[0]]) + self.pt_weighting(batch.pt[graph[1]])
        true_weights = weights[y].sum()
        fake_weights = weights[~y].sum()
        
        weights[y] = (weights[y]/true_weights)*torch.sigmoid(self.hparams["log_weight_ratio"]*torch.ones(1, device = self.device))
        weights[~y] = (weights[~y]/fake_weights)*torch.sigmoid(-self.hparams["log_weight_ratio"]*torch.ones(1, device = self.device))
        
        return weights.float()
    
    def get_asgmt_weight(self, batch, pt, bipartite_graph, y, row_match, col_match):
        
        supernodes_pt = torch.zeros(bipartite_graph[1].max() + 1, device = self.device).float()
        supernodes_pt[col_match] = pt[row_match].float()
        
        weights = torch.maximum(self.pt_weighting(batch.pt[bipartite_graph[0]]), self.pt_weighting(supernodes_pt[bipartite_graph[1]]))
        true_weights = weights[y].sum()
        fake_weights = weights[~y].sum()
        
        weights[y] = (weights[y]/true_weights)*torch.sigmoid(self.hparams["log_weight_ratio"]*torch.ones(1, device = self.device))
        weights[~y] = (weights[~y]/fake_weights)*torch.sigmoid(-self.hparams["log_weight_ratio"]*torch.ones(1, device = self.device))
        
        return weights.float()
    
    def get_hinge_distance(self, batch, embeddings, graph, y):
        
        hinge = torch.ones(len(y), device = self.device).long()
        hinge[~y] = -1
        
        dist = ((embeddings[graph[0]] - embeddings[graph[1]]).square().sum(-1)+1e-12).sqrt()
        
        return hinge, dist
        
    
    def training_step(self, batch, batch_idx):
       
        bipartite_graph, bipartite_scores, intermediate_embeddings = self(batch.x, batch.edge_index)
        
        # Compute embedding loss
        y_pid = batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
        weights = self.get_emb_weight(batch, batch.edge_index, y_pid)
        hinge, dist = self.get_hinge_distance(batch, intermediate_embeddings, batch.edge_index, y_pid)

        emb_loss = nn.functional.hinge_embedding_loss(dist, hinge, margin=self.hparams["train_r"], reduction='none').square()
        emb_loss = torch.dot(emb_loss, weights)
        

        # Convert PID&pT
        original_pid, pid = torch.unique(batch.pid, return_inverse = True)
        pt = scatter_min(batch.pt, pid, dim=0, dim_size = pid.max()+1)[0]
        
        # Matching objects
        with torch.no_grad():
            pid_cluster_mapping = csr_matrix(
                (torch.cat([bipartite_scores, 1e-12*torch.ones(pid.max()+1, device = self.device)], dim = 0).cpu().numpy(),
                (
                    torch.cat([pid[bipartite_graph[0]], torch.arange(pid.max()+1, device = self.device)], dim = 0).cpu().numpy(),
                    torch.cat([bipartite_graph[1], torch.arange(bipartite_graph[1].max()+1, bipartite_graph[1].max()+pid.max()+2, device = self.device)], dim = 0).cpu().numpy()
                )
                ),
                shape=(pid.max()+1, bipartite_graph[1].max()+pid.max()+2)
            )
            row_match, col_match = min_weight_full_bipartite_matching(pid_cluster_mapping, maximize=True)
            row_match, col_match = torch.tensor(row_match, device = self.device).long(), torch.tensor(col_match, device = self.device).long()
            noise_mask = (original_pid[row_match] != 0) & (col_match < bipartite_graph[1].max()+1)
            row_match, col_match = row_match[noise_mask], col_match[noise_mask]
        
            matched_particles = torch.tensor([False]*(pid.max()+1), device = self.device)
            matched_particles[row_match] = True
        
        # Compute bipartite loss
        pid_assignments = torch.zeros((pid.max()+1), device = self.device).long()
        pid_assignments[row_match] = col_match
        truth = torch.tensor([False]*len(bipartite_scores), device = self.device)
        matched_hits = matched_particles[pid[bipartite_graph[0]]]
        truth[matched_hits] = (pid_assignments[pid[bipartite_graph[0]][matched_hits]] == bipartite_graph[1][matched_hits])
        asgmt_loss = torch.nn.functional.binary_cross_entropy(bipartite_scores, truth.float(), reduction='none')
        asgmt_loss = torch.dot(asgmt_loss, self.get_asgmt_weight(batch, pt, bipartite_graph, truth, row_match, col_match))
        
        # Compute final loss using scheduling
        if "loss_schedule" in self.hparams and self.hparams["loss_schedule"] is not None:
            loss_schedule = self.hparams["loss_schedule"]
        else:
            loss_schedule = 1 - np.sin(self.trainer.current_epoch/2/self.hparams["emb_epoch"]*np.pi) if self.trainer.current_epoch < self.hparams["emb_epoch"] else 0
        loss = (loss_schedule * emb_loss) + ((1-loss_schedule)*asgmt_loss)
        
        self.log_dict(
            {
                "training_loss": loss,
                "embedding_loss": emb_loss,
                "assignment_loss": asgmt_loss
                
            }
        )

        
        return loss


    def shared_evaluation(self, batch, batch_idx, log=False):

        """
        This method is shared between validation steps and test steps
        """
        
        bipartite_graph, bipartite_scores, intermediate_embeddings = self(batch.x, batch.edge_index)
        
        # Compute embedding loss
        y_pid = batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]
        weights = self.get_emb_weight(batch, batch.edge_index, y_pid)
        hinge, dist = self.get_hinge_distance(batch, intermediate_embeddings, batch.edge_index, y_pid)

        emb_loss = nn.functional.hinge_embedding_loss(dist, hinge, margin=self.hparams["train_r"], reduction='none').square()
        emb_loss = torch.dot(emb_loss, weights)

        # Convert PID&pT
        original_pid, pid = torch.unique(batch.pid, return_inverse = True)
        pt = scatter_min(batch.pt, pid, dim=0, dim_size = pid.max()+1)[0]
        
        # Matching objects
        with torch.no_grad():
            pid_cluster_mapping = csr_matrix(
                (torch.cat([bipartite_scores, 1e-12*torch.ones(pid.max()+1, device = self.device)], dim = 0).cpu().numpy(),
                (
                    torch.cat([pid[bipartite_graph[0]], torch.arange(pid.max()+1, device = self.device)], dim = 0).cpu().numpy(),
                    torch.cat([bipartite_graph[1], torch.arange(bipartite_graph[1].max()+1, bipartite_graph[1].max()+pid.max()+2, device = self.device)], dim = 0).cpu().numpy()
                )
                ),
                shape=(pid.max()+1, bipartite_graph[1].max()+pid.max()+2)
            )
            row_match, col_match = min_weight_full_bipartite_matching(pid_cluster_mapping, maximize=True)
            row_match, col_match = torch.tensor(row_match, device = self.device).long(), torch.tensor(col_match, device = self.device).long()
            noise_mask = (original_pid[row_match] != 0) & (col_match < bipartite_graph[1].max()+1)
            row_match, col_match = row_match[noise_mask], col_match[noise_mask]
        
            matched_particles = torch.tensor([False]*(pid.max()+1), device = self.device)
            matched_particles[row_match] = True
        
        # Compute bipartite loss
        pid_assignments = torch.zeros((pid.max()+1), device = self.device).long()
        pid_assignments[row_match] = col_match
        truth = torch.tensor([False]*len(bipartite_scores), device = self.device)
        matched_hits = matched_particles[pid[bipartite_graph[0]]]
        truth[matched_hits] = (pid_assignments[pid[bipartite_graph[0]][matched_hits]] == bipartite_graph[1][matched_hits])
        asgmt_loss = torch.nn.functional.binary_cross_entropy(bipartite_scores, truth.float(), reduction='none')
        asgmt_loss = torch.dot(asgmt_loss, self.get_asgmt_weight(batch, pt, bipartite_graph, truth, row_match, col_match))
        
        if "loss_schedule" in self.hparams and isinstance(self.hparams["loss_schedule"], float):
            loss_schedule = self.hparams["loss_schedule"]
        else:
            if hasattr(self.trainer, "current_epoch") and self.training:
                loss_schedule = 1 - np.sin(self.trainer.current_epoch/2/self.hparams["emb_epoch"]*np.pi) if self.trainer.current_epoch < self.hparams["emb_epoch"] else 0
            else:
                loss_schedule = 0

        loss = (loss_schedule * emb_loss) + ((1-loss_schedule)*asgmt_loss)
        self.log_dict(
            {
                "val_loss": loss,
                "val_embedding_loss": emb_loss,
                "val_assignment_loss": asgmt_loss

            }
        )
        
        # Compute Tracking Efficiency
        bipartite_graph = bipartite_graph[:, bipartite_scores >= self.hparams["score_cut"]]
        bipartite_graph[0] = batch.inverse_mask[bipartite_graph[0]]
        event = torch.load(batch.dir[0], map_location=torch.device(self.device))
        if "1GeV" in str(batch.dir[0]):
            event = Data.from_dict(event.__dict__)
        event.pt[event.pid == 0] = 0
        _, inverse, counts = event.pid.unique(return_inverse = True, return_counts = True)
        event.nhits = counts[inverse]
        try:
            tracking_performace_metrics = eval_metrics(bipartite_graph,
                                                       event,
                                                       pt_cut = self.hparams["ptcut"],
                                                       nhits_cut = self.hparams["n_hits"],
                                                       majority_cut = self.hparams["majority_cut"],
                                                       primary = False)
        except:
            tracking_performace_metrics = {}
            
        
        if log:
            self.log_dict(
                {
                    **tracking_performace_metrics,
                }
            )
        return bipartite_graph, loss

    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=True)

        return outputs[1]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(batch, batch_idx, log=True)

        return outputs[1]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """
        Use this to manually enforce warm-up. In the future, this may become built-into PyLightning
        """
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.global_step < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                if self.hparams["model"] == "mlp" or self.hparams["model"] == 3:
                    pg["lr"] = lr_scale * self.hparams["mlp_lr"]
                else:
                    pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()