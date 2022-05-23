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
from torch_scatter import scatter_mean, scatter_add
from sklearn.metrics import roc_auc_score
import frnn

device = "cuda" if torch.cuda.is_available() else "cpu"

# Local imports
from .utils import load_dataset_paths, EmbeddingDataset, graph_intersection, build_graph

class GNNBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """
        self.save_hyperparameters(hparams)

    def setup(self, stage):
        
        # For now I only use single input which is the test set of the upstream pipeline
        paths = load_dataset_paths(self.hparams["input_dir"], self.hparams["datatype_names"])
        paths = paths[:sum(self.hparams["train_split"])]
        self.trainset, self.valset, self.testset = random_split(paths, self.hparams["train_split"], generator=torch.Generator().manual_seed(0))
        
    def train_dataloader(self):
        self.trainset = EmbeddingDataset(self.trainset, self.hparams, stage = "train", device = "cpu")
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=4)
        else:
            return None

    def val_dataloader(self):
        self.valset = EmbeddingDataset(self.valset, self.hparams, stage = "val", device = "cpu")
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=4)
        else:
            return None

    def test_dataloader(self):
        self.testset = EmbeddingDataset(self.testset, self.hparams, stage = "test", device = "cpu")
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

    
    def training_step(self, batch, batch_idx):
        
        input_data = self.get_input_data(batch)
        
        scores = self(input_data, batch.graph)
        
        truth_mask = (batch[self.hparams["truth_key"]] == 1)
        fake_mask = (batch.pid[batch.graph[0]] != batch.pid[batch.graph[1]]) | (batch.pid[batch.graph] == 0).any(0)
        
        mask = truth_mask | fake_mask
        
        scores = scores[mask]
        y = batch[self.hparams["truth_key"]][mask]
        
        weights = torch.ones(len(y), device = self.device)
        weights[y] = (1/(self.hparams["weight_ratio"] + 1)) * weights[y] / (weights[y].sum() + 1e-12)
        weights[~y] = (self.hparams["weight_ratio"]/(self.hparams["weight_ratio"] + 1)) * weights[~y] / (weights[~y].sum() + 1e-12)
        
        loss = nn.functional.binary_cross_entropy(scores, y.float(), reduction = "none")
        loss = torch.dot(weights, loss)
        
        self.log("training_loss", loss)
            
        return loss 
    
    def log_metrics(self, output, batch, loss, log):

        preds = output > self.hparams["score_cut"]

        # Positives
        edge_positive = preds.sum().float()
        
        # Signal true & tp
        sig_truth = batch.y
        sig_true = sig_truth.sum().float()
        sig_true_positive = (sig_truth.bool() & preds).sum().float()
        sig_auc = roc_auc_score(
            sig_truth.bool().cpu().detach(), output.cpu().detach()
        )

        # PID Signal true & signal tp
        sig_pid_truth = batch.pid_signal
        sig_pid_true = sig_pid_truth.sum().float()
        sig_pid_true_positive = (sig_pid_truth.bool() & preds).sum().float()
        sig_pid_auc = roc_auc_score(
            sig_pid_truth.bool().cpu().detach(), output.cpu().detach()
        )

        # Total true & total tp
        tot_truth = (batch.y_pid.bool() | batch.y.bool())
        tot_true = tot_truth.sum().float()
        tot_true_positive = (tot_truth.bool() & preds).sum().float()
        tot_auc = roc_auc_score(
            tot_truth.bool().cpu().detach(), output.cpu().detach()
        )

        # Eff, pur, auc
        sig_eff = sig_true_positive / sig_true
        sig_pur = sig_true_positive / edge_positive
        sig_pid_eff = sig_pid_true_positive / sig_pid_true
        sig_pid_pur = sig_pid_true_positive / edge_positive
        tot_eff = tot_true_positive / tot_true
        tot_pur = tot_true_positive / edge_positive

        # Combined metrics
        double_auc = sig_pid_auc * tot_auc
        custom_f1 = 2 * sig_pid_eff * tot_pur / (sig_pid_eff + tot_pur)
        sig_fake_ratio = sig_pid_true_positive / (edge_positive - tot_true_positive)

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {
                    "val_loss": loss,
                    "current_lr": current_lr,
                    "sig_eff": sig_eff,
                    "sig_pur": sig_pur,
                    "sig_auc": sig_auc,
                    "sig_pid_eff": sig_pid_eff,
                    "sig_pid_pur": sig_pid_pur,
                    "sig_pid_auc": sig_pid_auc,
                    "tot_eff": tot_eff,
                    "tot_pur": tot_pur,
                    "tot_auc": tot_auc,
                    "double_auc": double_auc,
                    "custom_f1": custom_f1,
                    "sig_fake_ratio": sig_fake_ratio,
                },
                sync_dist=True,
            )

        return preds

    def shared_evaluation(self, batch, batch_idx, log=False):

        """
        This method is shared between validation steps and test steps
        """
        
        input_data = self.get_input_data(batch)
        
        scores = self(input_data, batch.graph)
        
        truth_mask = (batch[self.hparams["truth_key"]] == 1)
        fake_mask = (batch.pid[batch.graph[0]] != batch.pid[batch.graph[1]]) | (batch.pid[batch.graph] == 0).any(0)
        
        mask = truth_mask | fake_mask
        
        loss_scores = scores[mask]
        y = batch[self.hparams["truth_key"]][mask]
        
        weights = torch.ones(len(y), device = self.device)
        weights[y] = (1/(self.hparams["weight_ratio"] + 1)) * weights[y] / (weights[y].sum() + 1e-12)
        weights[~y] = (self.hparams["weight_ratio"]/(self.hparams["weight_ratio"] + 1)) * weights[~y] / (weights[~y].sum() + 1e-12)
        
        loss = nn.functional.binary_cross_entropy(loss_scores, y.float(), reduction = "none")
        loss = torch.dot(weights, loss)
        
        eff = batch.y[scores > self.hparams["score_cut"]].sum()/batch.signal_true_edges.shape[1]
        pur = batch.y[scores > self.hparams["score_cut"]].sum()/(scores > self.hparams["score_cut"]).sum()
        pur_pid = batch.pid_signal[scores > self.hparams["score_cut"]].sum()/(scores > self.hparams["score_cut"]).sum()
        
        
        auc = roc_auc_score(y.cpu(), loss_scores.cpu())
            
        self.log_metrics(scores, batch, loss, log)
        
        return {
                    "val_loss": loss,
                    "eff": eff,
                    "pur": pur,
                    "pur_pid": pur_pid,
                    "auc": auc,
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