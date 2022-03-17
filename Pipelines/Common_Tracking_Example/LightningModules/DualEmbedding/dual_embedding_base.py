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

from sklearn.metrics import roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"

# Local imports
from .utils import load_dataset_paths, DualEmbeddingDataset, find_neighbors, graph_intersection

class DualEmbeddingBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """
        self.save_hyperparameters(hparams)
            
        self.cos = nn.CosineSimilarity()

    def setup(self, stage):
        
        # For now I only use single input which is the test set of the upstream pipeline
        input_dir = self.hparams["input_dir"]
        paths = load_dataset_paths(input_dir, sum(self.hparams["train_split"]))
        self.trainset, self.valset, self.testset = random_split(paths, self.hparams["train_split"], generator=torch.Generator().manual_seed(0))
        
    def train_dataloader(self):
        self.trainset = DualEmbeddingDataset(self.trainset, self.hparams, stage = "train", device = "cpu")
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=1, shuffle = True)
        else:
            return None

    def val_dataloader(self):
        self.valset = DualEmbeddingDataset(self.valset, self.hparams, stage = "val", device = "cpu")
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=1)
        else:
            return None

    def test_dataloader(self):
        self.testset = DualEmbeddingDataset(self.testset, self.hparams, stage = "test", device = "cpu")
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=1)
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
        
        if "ci" in self.hparams["regime"]:
            input_data = torch.cat([batch.cell_data[:, :self.hparams["cell_channels"]], batch.x], axis=-1)
        else:
            input_data = batch.x
            
        return input_data
    
    def pt_to_weight(self, pt):
        
        pt[pt!=pt] = 0

        h = lambda i: torch.heaviside(i, torch.zeros(1).to(pt))
        minimum = lambda i: torch.minimum(i, torch.ones(1).to(pt) - self.hparams["weight_min"])
        
        eps = self.hparams["weight_leak"]
        cut = self.hparams["signal_pt_cut"] - self.hparams["signal_pt_interval"]
        cap = self.hparams["signal_pt_cut"]
        
        return minimum((1 - self.hparams["weight_min"])*h(pt-cut)*(pt-cut)/(cap-cut)) + (eps * h(pt-cap) * (pt-cap)) + self.hparams["weight_min"]
    
    def get_hnm_negative_pairs(self, batch, embedding1, embedding2, radius):
        
        idxs = find_neighbors(embedding1.clone().detach(), embedding2.clone().detach(), r_max=radius, k_max=self.hparams["knn"])
        
        positive_idxs = idxs >= 0
        ind = torch.arange(idxs.shape[0], device = self.device).unsqueeze(1).expand(idxs.shape)
        edges = torch.stack([ind[positive_idxs],
                            idxs[positive_idxs]
                            ], dim = 0)
        edges = edges[:,batch.pid[edges[0]] != batch.pid[edges[1]]]
        return edges
    
    def training_step(self, batch, batch_idx):
        
        input_data = self.get_input_data(batch)
        
        embedding1, embedding2 = self(input_data) 
        
        with torch.no_grad():
            high_pt_edges = batch.true_edges[:, (batch.pt[batch.true_edges] > self.hparams["signal_pt_cut"]).any(0)]

            truth_d = torch.sqrt((embedding1[high_pt_edges[0]]
                                  - embedding2[high_pt_edges[1]]).square().sum(-1) + 1e-12)

            truth_d, _ = truth_d.sort()

            radius = truth_d[int(self.hparams["max_eff"]*len(truth_d))].item()
        
        fake_edges = self.get_hnm_negative_pairs(batch, embedding1, embedding2, radius)
        
        hinge = torch.cat([torch.ones(batch.true_edges.shape[1], device = self.device),
                          -torch.ones(fake_edges.shape[1], device = self.device)
                          ], dim = 0)
        
        edges = torch.cat([batch.true_edges,
                          fake_edges
                          ], dim = 1)
        
        weights = self.pt_to_weight(batch.pt)
        weights = weights[edges[0]] + weights[edges[1]]
        
        positive_weights = weights[hinge == 1].sum() + 1e-12
        negative_weights = weights[hinge == -1].sum() + 1e-12
        
        weights[hinge == 1] = self.hparams["weight_ratio"]*weights[hinge == 1]/positive_weights
        weights[hinge == -1] = weights[hinge == -1]/negative_weights
        weights = weights/(1 + self.hparams["weight_ratio"])
        
        
        if self.hparams["use_geodesic_distance"]:
            d = torch.acos((1 - 1e-4)*self.cos(embedding1[edges[0]], embedding2[edges[1]]))
            radius = torch.acos(torch.tensor(1 - 0.5*(radius**2), device = self.device))
        else:
            d = torch.sqrt((embedding1[edges[0]] - embedding2[edges[1]]).square().sum(-1) + 1e-12)
        
         
        loss = torch.nn.functional.hinge_embedding_loss(
            d/radius, hinge, margin=1, reduction="none"
        )

        loss = torch.dot(weights.to(loss), loss.square())
        
            
        self.log("train_loss", loss)

        return loss
    
    def shared_evaluation(self, batch, batch_idx, log=False):

        """
        This method is shared between validation steps and test steps
        """
        
        input_data = self.get_input_data(batch)
        embedding1, embedding2 = self(input_data)
        
        high_pt_edges = batch.true_edges[:, (batch.pt[batch.true_edges] > self.hparams["signal_pt_cut"]).any(0)]
        
        truth_d = torch.sqrt((embedding1[high_pt_edges[0]]
                              - embedding2[high_pt_edges[1]]).square().sum(-1) + 1e-12)
        
        truth_d, _ = truth_d.sort()
        
        radius = truth_d[int(self.hparams["max_eff"]*len(truth_d))].item()
        
        idxs = find_neighbors(embedding1.clone().detach(), embedding2.clone().detach(), r_max=radius, k_max=1000)
        
        positive_idxs = idxs >= 0
        ind = torch.arange(idxs.shape[0], device = self.device).unsqueeze(1).expand(idxs.shape)
        edges = torch.stack([ind[positive_idxs],
                            idxs[positive_idxs]
                            ], dim = 0)
        
        easy_fakes = edges[:,batch.pid[edges[0]]!=batch.pid[edges[1]]]
        e_ambiguous = edges[:,batch.pid[edges[0]]==batch.pid[edges[1]]]
        
        new_e_ambiguous, y = graph_intersection(e_ambiguous, batch.true_edges)
        
        y = torch.cat([torch.zeros(easy_fakes.shape[1], device = self.device), y.to(self.device)], dim = 0)
        edges = torch.cat([easy_fakes, new_e_ambiguous.to(self.device)], dim = 1)
        
        eff = (y.sum()/batch.true_edges.shape[1]).item()
        cut_eff = (y[(batch.pt[edges] > self.hparams["signal_pt_cut"]).all(0)].sum()/high_pt_edges.shape[1]).item()
        pur = (y.sum()/len(y)).item()
        cut_pur = (y[(batch.pt[edges] > self.hparams["signal_pt_cut"]).all(0)].sum()/len(y)).item()      
        
        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            
            self.log_dict(
                {
                    "dist@{}".format(self.hparams["max_eff"]): radius,
                    "eff": eff,
                    "cut_eff": cut_eff,
                    "pur": pur,
                    "cut_pur": cut_pur,
                    "current_lr": current_lr,
                }
            )
        return {
                "dist@{}".format(self.hparams["max_eff"]): radius,
                "eff": eff,
                "pur": pur,
                "cut_pur": cut_pur,
                "current_lr": current_lr
               }


    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=True)

        return outputs["pur"]

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