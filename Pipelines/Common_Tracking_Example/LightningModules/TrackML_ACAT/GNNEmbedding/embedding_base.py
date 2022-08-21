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


device = "cuda" if torch.cuda.is_available() else "cpu"

# Local imports
sys.path.append("../..")
from .utils import load_dataset_paths, EmbeddingDataset, FRNN_graph, graph_intersection
from tracking_utils import eval_metrics

class EmbeddingBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """
        self.save_hyperparameters(hparams)
        self.HDBSCANmodel = HDBSCAN(min_cluster_size = hparams["inference_min_cluster_size"], metric='euclidean', cluster_selection_method = "eom", verbose = 0)
        
    def setup(self, stage):
        
        # For now I only use single input which is the test set of the upstream pipeline
        paths = load_dataset_paths(self.hparams["input_dir"], self.hparams["datatype_names"])
        paths = paths[:sum(self.hparams["train_split"])]
        self.trainset, self.valset, self.testset = random_split(paths, self.hparams["train_split"], generator=torch.Generator().manual_seed(0))
        
    def train_dataloader(self):
        self.trainset = EmbeddingDataset(self.trainset, self.hparams, stage = "train", device = "cpu")
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=16, shuffle = True)
        else:
            return None

    def val_dataloader(self):
        self.valset = EmbeddingDataset(self.valset, self.hparams, stage = "val", device = "cpu")
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=16)
        else:
            return None

    def test_dataloader(self):
        self.testset = EmbeddingDataset(self.testset, self.hparams, stage = "test", device = "cpu")
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
    
    def get_training_weight(self, batch, graph, y):
        
        weights = self.pt_weighting(batch.pt[graph[0]]) + self.pt_weighting(batch.pt[graph[1]])
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
        
        embeddings, *_ = self(batch.x, batch.edge_index)
        
        graph, y = self.get_training_samples(embeddings, batch) 
        weights = self.get_training_weight(batch, graph, y)
        hinge, dist = self.get_hinge_distance(batch, embeddings, graph, y)
        
        loss = nn.functional.hinge_embedding_loss(dist, hinge, margin=self.hparams["train_r"], reduction='none').square()
        loss = torch.dot(loss, weights)
        
        self.log("training_loss", loss)
        
        return loss 


    def shared_evaluation(self, batch, batch_idx, log=False):

        """
        This method is shared between validation steps and test steps
        """
        
        embeddings, *_ = self(batch.x, batch.edge_index)
        
        # Compute Validation Loss
        graph, y = self.get_training_samples(embeddings, batch) 
        weights = self.get_training_weight(batch, graph, y)
        hinge, dist = self.get_hinge_distance(batch, embeddings, graph, y)
                     
        loss = nn.functional.hinge_embedding_loss(dist, hinge, margin=self.hparams["train_r"], reduction='none').square()
        loss = torch.dot(loss, weights)
        
        # Compute Tracking Efficiency        
        clusters = self.HDBSCANmodel.fit_predict(embeddings)
        clusters = torch.as_tensor(clusters).long().to(self.device)
        bipartite_graph = torch.stack([torch.arange(len(clusters), device = self.device), clusters], dim = 0)
        bipartite_graph = bipartite_graph[:, clusters >= 0]
        
        tracking_performace_metrics = eval_metrics(bipartite_graph,
                                                   batch,
                                                   pt_cut = self.hparams["ptcut"],
                                                   nhits_cut = self.hparams["n_hits"],
                                                   majority_cut = self.hparams["majority_cut"],
                                                   primary = False)
        
        if log:
            
            self.log_dict(
                {
                    **tracking_performace_metrics,
                    "val_loss": loss,
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