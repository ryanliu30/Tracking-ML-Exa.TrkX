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
from sklearn.cluster import KMeans
import frnn

device = "cuda" if torch.cuda.is_available() else "cpu"

# Local imports
from .utils import load_dataset_paths, EmbeddingDataset, graph_intersection, build_graph, efficiency_performance_wrt_distance

class GNNEmbeddingBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if hparams["use_toy"]:
            hparams["regime"] = []
            hparams["spatial_channels"] = 2

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
        self.trainset = EmbeddingDataset(self.trainset, self.hparams, stage = "train", device = "cpu")
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=16)
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
    
    def get_input_data(self, batch):
        
        input_data = batch.x
        
        if "ci" in self.hparams["regime"]:
            input_data = torch.cat([batch.cell_data[:, :self.hparams["cell_channels"]], input_data], axis=-1)
            
        return input_data
    
    def get_cluster(self, batch, embeddings):
        
        fake_embeddings = embeddings[batch.pid == 0]
        true_embeddings = embeddings[batch.pid != 0]

        true_clusters = batch.pid[batch.pid != 0]
        _, true_clusters = torch.unique(true_clusters, return_inverse = True)
            
        return fake_embeddings, true_embeddings, true_clusters
    
    def get_centers(self, true_embeddings, true_clusters):
        
        centers = torch.zeros((true_clusters.max()+1, self.hparams["emb_dim"]), device = self.device)
        centers = scatter_mean(true_embeddings, true_clusters, dim=0, out=centers)
        centers = nn.functional.normalize(centers, p=2.0, dim=1, eps=1e-12)
        
        return centers
    
    def attractive_potential(self, true_embeddings, fake_embeddings, true_clusters, centers):
        
        true_dist = (true_embeddings - centers[true_clusters]).square().sum(-1).mean()
        fake_dist = (fake_embeddings).square().sum(-1).mean()
        
        return true_dist, fake_dist
    
    def repulsive_potential(self, centers):
        
        edges = build_graph(centers, self.hparams["knn"], self.hparams["knn_r"])
        
        potential = torch.acos((1 - 1e-4)*self.cos(centers[edges[0]], centers[edges[1]]))
        potential = (self.hparams["knn_r"] - potential).square().mean()
        
        return potential
    
    def hard_negative_mining_loss(self, batch, embeddings):
        
        fake_embeddings = embeddings[batch.pid == 0]
        signal_embeddings = embeddings[(batch.pid != 0) & (batch.pt > self.hparams["ptcut"])]
        
        if fake_embeddings.shape[0] == 0:
            fake_embeddings = torch.zeros((1, self.hparams["emb_dim"]), device = self.device)
        
        mask = torch.arange(len(embeddings), device = self.device)[(batch.pid != 0) & (batch.pt > self.hparams["ptcut"])]
        
        hnm_edges = build_graph(signal_embeddings, self.hparams["knn"], self.hparams["knn_r"])
        hnm_edges = mask[hnm_edges]
        
        e_bidir = torch.cat([batch.signal_true_edges, batch.signal_true_edges.flip(0)], dim = -1)
        
        hnm_edges, y = graph_intersection(hnm_edges, e_bidir)
        hnm_edges = hnm_edges.to(self.device)
        y = y.to(self.device)

        fake_hnm_edges = hnm_edges[:, ((y == 0) & (batch.pid[hnm_edges[0]]!=batch.pid[hnm_edges[1]]))]
        dists = [(embeddings[e_bidir[0]] - embeddings[e_bidir[1]]).square().sum(-1).mean(),
                 fake_embeddings.square().sum(-1).mean(),
                 (self.hparams["knn_r"] - ((embeddings[fake_hnm_edges[0]] - embeddings[fake_hnm_edges[1]]).square().sum(-1) + 1e-12).sqrt()).square().mean(),
                 (1 - (signal_embeddings.square().sum(-1)+1e-12).sqrt()).square().mean()
                ]
        
        loss = sum([weight*dist for weight, dist in zip(self.hparams["hnm_weights"], dists)])/sum(self.hparams["hnm_weights"])
        
        return loss
    
    
    def training_step(self, batch, batch_idx):
        
        input_data = self.get_input_data(batch)
        
        embeddings = self(input_data, batch.graph)
        
        if self.hparams["loss_function"] == "object_condensation":
        
            fake_embeddings, true_embeddings, true_clusters = self.get_cluster(batch, embeddings)
            centers = self.get_centers(true_embeddings, true_clusters)
            dists = [*self.attractive_potential(true_embeddings, fake_embeddings, true_clusters, centers), self.repulsive_potential(centers)]
            loss = sum([weight*dist for weight, dist in zip(self.hparams["weights"], dists)])/sum(self.hparams["weights"])
            
        if self.hparams["loss_function"] == "hard_negative_mining":
            
            loss = self.hard_negative_mining_loss(batch, embeddings)
        
        self.log("embedding_loss", loss)
            
        return loss 
    

    def shared_evaluation(self, batch, batch_idx, log=False):

        """
        This method is shared between validation steps and test steps
        """
        
        input_data = self.get_input_data(batch)
        
        embeddings= self(input_data, batch.graph)
            
        if self.hparams["loss_function"] == "object_condensation":
            
            fake_embeddings, true_embeddings, true_clusters = self.get_cluster(batch, embeddings)
            centers = self.get_centers(true_embeddings, true_clusters)
            dists = [*self.attractive_potential(true_embeddings, fake_embeddings, true_clusters, centers), self.repulsive_potential(centers)]
            loss = sum([weight*dist for weight, dist in zip(self.hparams["weights"], dists)])/sum(self.hparams["weights"])
            
        if self.hparams["loss_function"] == "hard_negative_mining":
            loss = self.hard_negative_mining_loss(batch, embeddings)
            
        # On top of old graph
            
        dist = (embeddings[batch.graph[0]] - embeddings[batch.graph[1]]).square().sum(-1).sqrt()
        y = batch.y.clone()
        
        truth_dist = (embeddings[batch.signal_true_edges[0]] - embeddings[batch.signal_true_edges[1]]).square().sum(-1).sqrt()
        cut_dist = truth_dist.sort()[0][int(self.hparams["max_eff"]*len(truth_dist))].clone().detach().item()
        
        positives = (dist[batch.scores >= self.hparams["score_cut"]] < cut_dist).sum()
        true_positives = (dist[batch.scores >= self.hparams["score_cut"]] < cut_dist)[y[batch.scores >= self.hparams["score_cut"]] == 1].sum()

        true = batch.signal_true_edges.shape[1]
        original_pur = batch.y[batch.scores >= self.hparams["score_cut"]].sum()/len(batch.y[batch.scores >= self.hparams["score_cut"]])
            
        eff = true_positives/true
        pur = true_positives/positives
        pur_boost = pur/original_pur
        
        # Construct New Graph
        
        new_graph = build_graph(embeddings, 1000, cut_dist)
                        
        e_bidir = torch.cat([batch.signal_true_edges, batch.signal_true_edges.flip(0)], dim = -1)
        new_graph, y = graph_intersection(new_graph, e_bidir)
        new_pur = y.sum()/new_graph.shape[1]
        new_eff = y.sum()/e_bidir.shape[1]
        
        pid_eff = (batch.pid[new_graph[0]] == batch.pid[new_graph[1]]).sum()/(batch.pid.unique(return_counts = True)[1]*(batch.pid.unique(return_counts = True)[1]-1)).sum()
        pid_pur = (batch.pid[new_graph[0]] == batch.pid[new_graph[1]]).sum()/new_graph.shape[1]
        
        n_hop_eff_list, n_hop_pur_list = efficiency_performance_wrt_distance(batch.graph, new_graph, batch.signal_true_edges, 5)
        
        n_hop_eff = {"{}_hop_eff".format(i+1): n_hop_eff_list[i] for i in range(len(n_hop_eff_list))}
        n_hop_pur = {"{}_hop_pur".format(i+1): n_hop_pur_list[i] for i in range(len(n_hop_pur_list))}
        
        if log:
            
            current_lr = self.optimizers().param_groups[0]["lr"]
            
            self.log_dict(
                {
                    "val_loss": loss,
                    "eff": eff,
                    "pur": pur,
                    "pid_eff":pid_eff,
                    "pid_pur":pid_pur,
                    "new_eff": new_eff,
                    "new_pur": new_pur,
                    "pur_boost": pur_boost,
                    "current_lr": current_lr,
                    **n_hop_eff,
                    **n_hop_pur
                    
                }
            )
        return {
                "val_loss": loss,
                "eff": eff,
                "pur": pur,
                "pid_eff":pid_eff,
                "pid_pur":pid_pur,
                "new_eff": new_eff,
                "new_pur": new_pur,
                "pur_boost": pur_boost,
                "current_lr": current_lr,
                **n_hop_eff,
                **n_hop_pur
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