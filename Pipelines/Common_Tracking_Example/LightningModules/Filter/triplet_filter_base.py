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

from sklearn.metrics import roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"

# Local imports
from .utils import graph_intersection, load_dataset_paths, filter_dataset

class FilterBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """
        self.save_hyperparameters(hparams)
        
        if "logger" in self.__dict__.keys() and "_experiment" in self.logger.__dict__.keys():
            self.logger.experiment.define_metric("val_loss" , summary="min")
            self.logger.experiment.define_metric("auc" , summary="max")

    def setup(self, stage):
        # Handle any subset of [train, val, test] data split, assuming that ordering
        
        input_dirs = [None for i in self.hparams["datatype_names"]]
        input_dirs[: len(self.hparams["datatype_names"])] = [
            os.path.join(self.hparams["input_dir"], datatype)
            for datatype in self.hparams["datatype_names"]
        ]
        paths = load_dataset_paths(input_dirs, sum(self.hparams["datatype_split"]))
        self.trainset, self.valset, self.testset = random_split(paths, self.hparams["datatype_split"], generator=torch.Generator().manual_seed(0))
        
    def train_dataloader(self):
        self.trainset = filter_dataset(self.trainset, self.hparams, stage = "train")
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=8)
        else:
            return None

    def val_dataloader(self):
        self.valset = filter_dataset(self.valset, self.hparams, stage = "val")
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=1)
        else:
            return None

    def test_dataloader(self):
        self.testset = filter_dataset(self.testset, self.hparams, stage = "test")
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
    
    def pt_to_weight(self, pt):
        
        pt[pt!=pt] = 0

        h = lambda i: torch.heaviside(i, torch.zeros(1).to(pt))
        minimum = lambda i: torch.minimum(i, torch.ones(1).to(pt))
        
        eps = self.hparams["weight_leak"]
        cut = self.hparams["signal_pt_cut"] - self.hparams["signal_pt_interval"]
        cap = self.hparams["signal_pt_cut"]
        
        return minimum(h(pt-cut)*(pt-cut)/(cap-cut)) + (eps * h(pt-cap) * (pt-cap))
    
    def pt_to_weight_fake(self, pt):
        
        pt[pt!=pt] = 0

        h = lambda i: torch.heaviside(i, torch.zeros(1).to(pt))
        maximum = lambda i: torch.maximum(i, torch.ones(1).to(pt))
        
        intercept = self.hparams["intercept"]
        cap = self.hparams["signal_pt_cut"]
        
        return maximum(intercept - (intercept - 1)*pt/cap)
    
    def get_edge_weight(self, e_spatial, y_cluster, pt):
        
        weight = torch.zeros(y_cluster.shape).to(self.device)
        
        weight[y_cluster != 0] = (self.pt_to_weight(pt[e_spatial[0][y_cluster != 0]]) 
                                  + self.pt_to_weight(pt[e_spatial[1][y_cluster != 0]])).to(weight)
        weight[y_cluster == 0] = (self.pt_to_weight_fake(pt[e_spatial[0][y_cluster == 0]]) 
                                  + self.pt_to_weight_fake(pt[e_spatial[1][y_cluster == 0]])).to(weight)
        
        unnormalized_weight = weight.clone().detach()
        
        truth_weight = weight[y_cluster != 0].sum() + 1e-12
        fake_weight = weight[y_cluster == 0].sum() + 1e-12
        
        if (y_cluster != 0).any(0):
            weight[y_cluster != 0] = self.hparams["weight_ratio"] * weight[y_cluster != 0] / truth_weight
        if (y_cluster == 0).any(0):
            weight[y_cluster == 0] = weight[y_cluster == 0] / fake_weight
        
        return weight/(1+self.hparams["weight_ratio"])*len(y_cluster), unnormalized_weight

    def get_input_data(self, batch):
        
        if "ci" in self.hparams["regime"]:
            input_data = torch.cat([batch.cell_data[:, :self.hparams["cell_channels"]], batch.x], axis=-1)
            input_data[input_data != input_data] = 0
        else:
            input_data = batch.x
            input_data[input_data != input_data] = 0
            
        return input_data
    
    def training_step(self, batch, batch_idx):
        
        input_data = self.get_input_data(batch)

        ind = torch.Tensor.repeat(torch.arange(batch.idxs.shape[0], device=device), (batch.idxs.shape[1], 1)).T.int()

        positive_idxs = batch.idxs >= 0
        edges = torch.stack([ind[positive_idxs], batch.idxs[positive_idxs]]).long()

        edge_samples = edges[:,torch.randint(0, edges.shape[1], (self.hparams["edges_per_batch"],), device = self.device)].to(self.device)
        fake_mask = ((batch.pid[edge_samples[0]] == batch.pid[edge_samples[1]]) & (batch.pid[edge_samples] != 0).all(0)).logical_not()
        edge_samples = edge_samples[:,fake_mask]
        
        e_bidir = torch.cat(
            [batch.modulewise_true_edges, batch.modulewise_true_edges.flip(0)], axis=-1
        )
        edges = torch.cat([
                        edge_samples,
                        e_bidir.to(self.device),
                    ], dim = 1)
    
        y = torch.cat([torch.zeros(edge_samples.shape[1]), torch.ones(e_bidir.shape[1])], dim = 0).to(self.device)
        weights, _ = self.get_edge_weight(edges, y, batch.pt)
        
        output = self(
                    input_data,
                    edges
                ).squeeze()
        
        loss = F.binary_cross_entropy_with_logits(
                output,
                y.float(),
                weight=weights,
                reduction = "mean",
            )
            
        self.log("train_loss", loss)

        return loss
    
    def shared_evaluation(self, batch, batch_idx, log=False):

        """
        This method is shared between validation steps and test steps
        """
        
        score_list = []
        val_loss = torch.tensor(0).to(self.device)
        chunks = torch.chunk(batch.idxs, self.hparams["n_chunks"], dim = 1)
        input_data = self.get_input_data(batch)
        e_bidir = torch.cat(
            [batch.modulewise_true_edges, batch.modulewise_true_edges.flip(0)], axis=-1
        )
        all_edges = torch.empty([2, 0], dtype=torch.int64).cpu()
        all_y = torch.empty([0], dtype=torch.int64).cpu()
        all_scores = torch.empty([0], dtype=torch.float).cpu()
        all_weights = torch.empty([0], dtype=torch.float).cpu()
        
        for chunk in chunks:

            scores = torch.zeros(chunk.shape).to(self.device)
            ind = torch.Tensor.repeat(torch.arange(chunk.shape[0], device=device), (chunk.shape[1], 1)).T.int()
            
            positive_idxs = chunk >= 0
            edges = torch.stack([ind[positive_idxs], chunk[positive_idxs]]).long()
            
            output = self(
                    input_data,
                    edges
                ).squeeze()
            scores[positive_idxs] = torch.sigmoid(output)
            score_list.append(scores.detach().cpu())
            
            # compute val loss
            truth_mask = (batch.pid[edges[0]] == batch.pid[edges[1]]) & (batch.pid[edges] != 0).all(0)
            edges_easy_fake = edges[:,truth_mask.logical_not()].clone().detach()
            edges_ambiguous = edges[:,truth_mask].clone().detach()
            if edges_ambiguous.numel() != 0:
                edges_ambiguous, y_ambiguous = graph_intersection(edges_ambiguous, e_bidir)
                edges = torch.cat([edges_easy_fake, edges_ambiguous.to(self.device)], dim = 1)
                y = torch.cat([torch.zeros(edges_easy_fake.shape[1]), y_ambiguous], dim = 0)
            else: 
                edges = edges_easy_fake
                y = torch.zeros(edges_easy_fake.shape[1])
            
            weights, _ = self.get_edge_weight(edges, y, batch.pt)
            
            output = self(
                    input_data,
                    edges
                ).squeeze()
            
            all_weights = torch.cat([all_weights, weights.cpu()], dim = 0)
            all_scores = torch.cat([all_scores, torch.sigmoid(output).cpu()], dim = 0)
            all_edges = torch.cat([all_edges, edges.cpu()], dim = 1)
            all_y = torch.cat([all_y, y.cpu()], dim = 0)
            
            val_loss = val_loss + F.binary_cross_entropy_with_logits(
                    output, y.float().to(self.device), weight = weights
                )
            
        score_list = torch.cat(score_list, dim = 1)
        
        
        # Find Truth Scores
        pt_mask = (batch.pt[e_bidir] >= self.hparams["signal_pt_cut"]).all(0)
        output = self(
                    input_data,
                    e_bidir[:, pt_mask]
                ).squeeze()
        scores = torch.sigmoid(output)
        scores, _ = scores.sort(descending=True)
        eff_cut_score = scores[int(self.hparams["max_eff"]*len(scores))]
        
        cut_list = (all_scores >= eff_cut_score.item())
        
        # For efficeincy and purity, evaluate on modulewise truth.
        modulewise_true = pt_mask.sum()
        prediction_pt_mask = (batch.pt[all_edges] >= self.hparams["signal_pt_cut"]).all(0)
        modulewise_true_positive = (all_y.bool() & cut_list)[prediction_pt_mask].sum()
        modulewise_true_positive_without_cut = (all_y.bool() & cut_list).sum()
        modulewise_positive = cut_list.sum()
        
        # For auc, again use modulewise truth
        auc = roc_auc_score(all_y.bool().cpu().detach(),
                            all_scores.cpu().detach())

        if log:
            
            current_lr = self.optimizers().param_groups[0]["lr"]
            
            self.log_dict(
                {
                    "score@{}".format(self.hparams["max_eff"]): eff_cut_score.item(),
                    "eff": (modulewise_true_positive / modulewise_true).clone().detach(),
                    "pur": (modulewise_true_positive_without_cut / modulewise_positive).clone().detach(),
                    "val_loss": val_loss/self.hparams["n_chunks"],
                    "current_lr": current_lr,
                    "auc": auc
                }
            )
        return {"loss": val_loss, "preds": score_list}


    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=True)

        return outputs["loss"]

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