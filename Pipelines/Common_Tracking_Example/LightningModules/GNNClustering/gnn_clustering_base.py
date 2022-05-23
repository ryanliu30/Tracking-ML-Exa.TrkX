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
import wandb
from torch.utils.data import random_split
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add, scatter_min
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

device = "cuda" if torch.cuda.is_available() else "cpu"

# Local imports
from .utils import load_dataset_paths, ClusteringDataset

class GNNClusteringBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """
        self.save_hyperparameters(hparams)
        self.cos = nn.CosineSimilarity()

    def setup(self, stage):
        
        # For now I only use single input which is the test set of the upstream pipeline
        paths = load_dataset_paths(self.hparams["input_dir"], self.hparams["datatype_names"])
        paths = paths[:sum(self.hparams["train_split"])]
        self.trainset, self.valset, self.testset = random_split(paths, self.hparams["train_split"], generator=torch.Generator().manual_seed(0))
        
    def train_dataloader(self):
        self.trainset = ClusteringDataset(self.trainset, self.hparams, stage = "train", device = "cpu")
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=4)
        else:
            return None

    def val_dataloader(self):
        self.valset = ClusteringDataset(self.valset, self.hparams, stage = "val", device = "cpu")
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=4)
        else:
            return None

    def test_dataloader(self):
        self.testset = ClusteringDataset(self.testset, self.hparams, stage = "test", device = "cpu")
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=4)
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
    
    def get_input_data(self, batch):
        
        input_data = batch.x
        
        if "ci" in self.hparams["regime"]:
            input_data = torch.cat([batch.cell_data[:, :self.hparams["cell_channels"]], input_data], axis=-1)
            
        return input_data
    
    def pt_weighting(self, pt):
        
        pt[pt!=pt] = 0

        h = lambda i: torch.heaviside(i, torch.zeros(1).to(pt))
        minimum = lambda i: torch.minimum(i, torch.ones(1).to(pt))
        
        eps = self.hparams["weight_leak"]
        cut = self.hparams["ptcut"] - self.hparams["pt_interval"]
        cap = self.hparams["ptcut"]
        min_weight = self.hparams["weight_min"]
        
        return min_weight + (1-min_weight)*minimum(h(pt-cut)*(pt-cut)/(cap-cut)) + (eps * h(pt-cap) * (pt-cap))
    
    
    def training_step(self, batch, batch_idx):
        
        input_data = self.get_input_data(batch)
        
        assignments, unnormalized_assignment = self(input_data, batch.assignment_init, batch.graph)
                     
        _, pid = torch.unique(batch.pid.long(), return_inverse = True)
        pt = scatter_min(batch.pt, pid, dim=0, dim_size = pid.max()+1)[0]
        matching = torch.zeros(pid.max()+1, device = self.device).long()
        
        assignments = scatter_add(assignments, pid, dim = 0, dim_size = pid.max()+1)
        probability = assignments/assignments.sum(1).unsqueeze(1)
        signal_sample = (pt > self.hparams["ptcut"])
        inverse_mask = torch.arange(len(signal_sample), device = self.device)[signal_sample]
        
        bipartite_graph = csr_matrix(probability[signal_sample].clone().detach().cpu().numpy())
        row_match, col_match = min_weight_full_bipartite_matching(bipartite_graph, maximize=True)
        matching[row_match] = torch.tensor(col_match, device = self.device).long()
        
        if (~signal_sample).any():
            assignments[:,col_match] = -1
            matching[~signal_sample] = assignments[~signal_sample].argmax(1)
        
        labels = matching[pid]
        loss = nn.functional.cross_entropy(unnormalized_assignment, labels, reduction = "none")
        
        weights = self.pt_weighting(batch.pt)
        weights = weights/weights.sum()
        
        loss = torch.dot(loss, weights)
                     
        self.log("training_loss", loss)
            
        return loss 


    def shared_evaluation(self, batch, batch_idx, log=False):

        """
        This method is shared between validation steps and test steps
        """
        
        input_data = self.get_input_data(batch)
        
        assignments, unnormalized_assignment = self(input_data, batch.assignment_init, batch.graph)
        original_assignments = assignments.clone()
                     
        _, pid = torch.unique(batch.pid.long(), return_inverse = True)
        pt = scatter_min(batch.pt, pid, dim=0, dim_size = pid.max()+1)[0]
        matching = torch.zeros(pid.max()+1, device = self.device).long()
        
        assignments = scatter_add(assignments, pid, dim = 0, dim_size = pid.max()+1)
        probability = assignments/(assignments.sum(1).unsqueeze(1) + 1e-12)
        signal_sample = (pt > self.hparams["ptcut"])
        inverse_mask = torch.arange(len(signal_sample), device = self.device)[signal_sample]
        
        bipartite_graph = csr_matrix(probability[signal_sample].clone().cpu().numpy())
        row_match, col_match = min_weight_full_bipartite_matching(bipartite_graph, maximize=True)
        matching[row_match] = torch.tensor(col_match, device = self.device).long()
        
        if (~signal_sample).any():
            assignments[:,col_match] = -1
            matching[~signal_sample] = assignments[~signal_sample].argmax(1)
        
        labels = matching[pid]
        loss = nn.functional.cross_entropy(unnormalized_assignment, labels, reduction = "none")
        
        weights = self.pt_weighting(batch.pt)
        weights = weights/weights.sum()
        
        loss = torch.dot(loss, weights) 
        
        labels = original_assignments.argmax(1)
        labels = nn.functional.one_hot(labels, num_classes= self.hparams["clusters"]).float()
        pid_cluster_counts = scatter_add(labels, pid, dim = 0, dim_size = pid.max()+1)
        bipartite_matrix = csr_matrix(scatter_add(original_assignments, pid, dim = 0, dim_size = pid.max()+1).cpu().numpy())
        row_match, col_match = min_weight_full_bipartite_matching(bipartite_matrix, maximize=True)
        
        majority_mask = (pid_cluster_counts[row_match, col_match]/pid_cluster_counts[:, col_match].sum(0) > self.hparams["majority_cut"])
        pt_mask = (pt[row_match] > self.hparams["ptcut"])
        
        eff = (majority_mask & pt_mask).sum()/(pt > self.hparams["ptcut"]).sum()
        pur = (pid_cluster_counts[row_match, col_match]/pid_cluster_counts[:, col_match].sum(0))[majority_mask].mean()
        
        if log:
            
            current_lr = self.optimizers().param_groups[0]["lr"]
            
            self.log_dict(
                {
                    "pur": pur,
                    "eff": eff,
                    "val_loss": loss,
                }
            )
        return {
                    "pur": pur,
                    "eff": eff,
                    "assignments": assignments,
                    "val_loss": loss,
               }

    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=True)

        return outputs["val_loss"]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(batch, batch_idx, log=False)

        return outputs

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
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()