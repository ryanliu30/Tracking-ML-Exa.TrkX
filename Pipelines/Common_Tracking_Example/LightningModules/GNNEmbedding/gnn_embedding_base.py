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
from .utils import load_dataset_paths, EmbeddingDataset, graph_intersection

class GNNEmbeddingBase(LightningModule):
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
        
        if "ci" in self.hparams["regime"]:
            input_data = torch.cat([batch.cell_data[:, :self.hparams["cell_channels"]], batch.x], axis=-1)
        else:
            input_data = batch.x
            
        return input_data
    
    def get_cluster(self, batch, embeddings, embedding_regime = "node"):
        
        if embedding_regime == "node":
            fake_embeddings = embeddings[batch.pid == 0]
            true_embeddings = embeddings[batch.pid != 0]
            
            true_clusters = batch.pid[batch.pid != 0]
            _, true_clusters = torch.unique(true_clusters, return_inverse = True)
        
        if embedding_regime == "edge":
            
            fake_embeddings = embeddings[batch.y_pid == 0]
            true_embeddings = embeddings[batch.y_pid != 0]
            
            true_clusters_start = batch.pid[batch.graph[:,batch.y_pid != 0][0]]
            true_clusters_end = batch.pid[batch.graph[:,batch.y_pid != 0][1]]
            true_clusters_different = (true_clusters_start != true_clusters_end)
            
            true_clusters = torch.cat([true_clusters_start,
                                       true_clusters_end[true_clusters_different]], dim = 0)
            true_embeddings = torch.cat([true_embeddings,
                                         true_embeddings[true_clusters_different]], dim = 0)
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
        
        return true_dist + fake_dist
    
    def repulsive_potential(self, centers):
        
        centers = centers.unsqueeze(0)
        _, idxs, _, _ = frnn.frnn_grid_points(points1 = centers, points2 = centers, K = 100, r = self.hparams["knn_r"])
        
        idxs = idxs.squeeze()
        centers = centers.squeeze()
        positive_idxs = idxs >= 0
        ind = torch.arange(len(centers), device = self.device).reshape((-1, 1)).expand((-1, 100))
        edges = torch.stack([ind[positive_idxs],
                            idxs[positive_idxs]
                            ], dim = 0)
        
        potential = torch.acos((1 - 1e-4)*self.cos(centers[edges[0]], centers[edges[1]]))
        potential = (self.hparams["knn_r"] - potential).square().mean()
        
        return potential
    
    def get_hinge_loss(self, batch, embeddings):
        
        if self.hparams["embedding_regime"] == "node":
            
            dist = torch.acos((1 - 1e-4)*self.cos(embeddings[batch.graph[0]], embeddings[batch.graph[1]]))
            hinge = batch.y.clone()
            mask = (batch.y | torch.logical_not(batch.y_pid)) & (batch.pid[batch.graph] != 0).all(0)
            
        if self.hparams["embedding_regime"] == "edge":
        
            dist = torch.acos((1 - 1e-4)*self.cos(embeddings[batch.triplet_graph[0]], embeddings[batch.triplet_graph[1]]))
            hinge = batch.triplet_y.clone()
            mask = (batch.triplet_y | torch.logical_not(batch.triplet_y_pid)) & (batch.y[event.triplet_graph] != 0).all(0)
            
        hinge[hinge == 0] = -1
        
        return dist[mask], hinge[mask]
    
    def get_radial_loss(self, batch, embeddings):
        
        norm = embeddings.square().sum(-1).sqrt()
        
        if self.hparams["embedding_regime"] == "node":
            
            hinge = batch.pid
            hinge[hinge > 0] = 1
            hinge[hinge == 0] = -1
            
        if self.hparams["embedding_regime"] == "edge":
            
            hinge = batch.y_pid.clone()
            hinge[hinge == 0] = -1
        
        loss = nn.functional.hinge_embedding_loss(norm, hinge, margin=1, reduction="none")
        loss = loss.square().mean()
        
        return loss
        
    
    def get_dist(self, batch, embeddings):
        
        if self.hparams["embedding_regime"] == "node":
            
            dist = (embeddings[batch.graph[0]] - embeddings[batch.graph[1]]).square().sum(-1).sqrt()
            y = batch.y.clone()
            
        if self.hparams["embedding_regime"] == "edge":
        
            dist = (embeddings[batch.triplet_graph[0]] - embeddings[batch.triplet_graph[1]]).square().sum(-1).sqrt()
            y = batch.triplet_y.clone()

        return dist, y
    
    def training_step(self, batch, batch_idx):
        
        input_data = self.get_input_data(batch)
        
        embeddings = self(input_data, batch.graph)
        
        if self.hparams["loss_function"] == "object_condensation":
            
            fake_embeddings, true_embeddings, true_clusters = self.get_cluster(batch, embeddings, embedding_regime = self.hparams["embedding_regime"])
            centers = self.get_centers(true_embeddings, true_clusters)
            loss = (self.attractive_potential(true_embeddings, fake_embeddings, true_clusters, centers) + 
                    self.repulsive_potential(centers))
            
        if self.hparams["loss_function"] == "pairwise_truth":
            
            dist, hinge = self.get_hinge_loss(batch, embeddings)
            loss = nn.functional.hinge_embedding_loss(dist, hinge, margin=1, reduction="none")
            loss = loss.square().mean() + self.get_radial_loss(batch, embeddings)
                
                
        self.log("train_loss", loss)

        return loss
    

    def shared_evaluation(self, batch, batch_idx, log=False):

        """
        This method is shared between validation steps and test steps
        """
        
        input_data = self.get_input_data(batch)
        
        embeddings = self(input_data, batch.graph)
        
        if self.hparams["loss_function"] == "object_condensation":
            
            fake_embeddings, true_embeddings, true_clusters = self.get_cluster(batch, embeddings, embedding_regime = self.hparams["embedding_regime"])
            centers = self.get_centers(true_embeddings, true_clusters)
            loss = (self.attractive_potential(true_embeddings, fake_embeddings, true_clusters, centers) + self.repulsive_potential(centers))        
        
        if self.hparams["loss_function"] == "pairwise_truth":
            
            dist, hinge = self.get_hinge_loss(batch, embeddings)
            loss = nn.functional.hinge_embedding_loss(dist, hinge, margin=self.hparams["margin"], reduction="none")
            loss = loss.square().mean() + self.get_radial_loss(batch, embeddings)
        
        dist, y = self.get_dist(batch, embeddings)
        
        scores = (dist.max() - dist)/dist.max()
        auc = roc_auc_score(y.cpu().numpy(), scores.cpu().numpy())
        
        true_dist = dist[y == 1]
        true_dist, _ = true_dist.sort()
        
        cut_dist = true_dist[int(self.hparams["max_eff"]*len(true_dist))]
        
        positives = (dist < cut_dist).sum()
        true_positives = (true_dist < cut_dist).sum()
        
        if self.hparams["embedding_regime"] == "node":
            true = batch.signal_true_edges.shape[1]
            original_pur = batch.y.sum()/len(batch.y)
        if self.hparams["embedding_regime"] == "edge":
            true = batch.signal_triplet_true_graph.shape[1]
            original_pur = batch.triplet_y.sum()/len(batch.triplet_y)
            
        eff = true_positives/true
        pur = true_positives/positives
        pur_boost = pur/original_pur
        
        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            
            self.log_dict(
                {
                    "val_loss": loss,
                    "dist@{}".format(self.hparams["max_eff"]): cut_dist,
                    "eff": eff,
                    "pur": pur,
                    "pur_boost": pur_boost,
                    "auc": auc,
                    "current_lr": current_lr,
                }
            )
        return {
                "val_loss": loss,
                "dist@{}".format(self.hparams["max_eff"]): cut_dist,
                "eff": eff,
                "pur": pur,
                "pur_boost": pur_boost,
                "auc": auc,
                "current_lr": current_lr
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