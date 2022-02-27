"""The base classes for the embedding process.

The embedding process here is both the embedding models (contained in Models/) and the training procedure, which is a Siamese network strategy. Here, the network is run on all points, and then pairs (or triplets) are formed by one of several strategies (e.g. random pairs (rp), hard negative mining (hnm)) upon which some sort of contrastive loss is applied. The default here is a hinge margin loss, but other loss functions can work, including cross entropy-style losses. Also available are triplet loss approaches.

Example:
    See Quickstart for a concrete example of this process.
    
Todo:
    * Refactor the training & validation pipeline, since the use of the different regimes (rp, hnm, etc.) looks very messy
"""

# System imports
import sys
import os
import logging

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.nn import Linear
from torch_geometric.data import DataLoader
from torch_cluster import radius_graph
import numpy as np
import matplotlib.pyplot as plt
import random

# Local Imports
from .utils import graph_intersection, split_datasets, multi_build_edges

device = "cuda" if torch.cuda.is_available() else "cpu"


class SphereEmbeddingBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)
        self.cos = nn.CosineSimilarity()

    def setup(self, stage):
        self.trainset, self.valset, self.testset = split_datasets(
            **self.hparams
        )

    def train_dataloader(self):
        if len(self.trainset) > 0:
            return DataLoader(self.trainset, batch_size=1, num_workers=1)
        else:
            return None

    def val_dataloader(self):
        if len(self.valset) > 0:
            return DataLoader(self.valset, batch_size=1, num_workers=1)
        else:
            return None

    def test_dataloader(self):
        if len(self.testset):
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

    def get_input_data(self, batch, data_type = "signal_"):
        
        if "ci" in self.hparams["regime"]:
            input_data = torch.cat([batch[data_type + "cell_data"][:, :self.hparams["cell_channels"]], batch[data_type + "x"]], axis=-1)
            
        else:
            input_data = batch[data_type + "x"]
            
        return input_data
    
    def get_query_points(self, batch, spatials):
        
        query_indices = batch.signal_true_edges.unique()
        query_indices = query_indices[torch.randperm(len(query_indices))][:self.hparams["signal_points_per_batch"]]
        
        queries = []
        for spatial in spatials:
            queries.append(spatial[query_indices])
        
        return query_indices, queries
    
    def append_hnm_pairs(self, e_spatial, query, query_indices, spatial):
        knn_edges = build_edges(query, spatial, query_indices, self.hparams["r_train"], self.hparams["knn"])
        e_spatial = torch.cat(
            [
                e_spatial,
                knn_edges,
            ],
            axis=-1,
        )
        return e_spatial
    
    def multi_append_hnm_pairs(self, e_spatial, query, query_indices, spatial):
        knn_edges = multi_build_edges(query, spatial, query_indices, self.hparams["r_train"], int(self.hparams["knn"]/(self.hparams["n_spaces"] + 1)))
        e_spatial = torch.cat(
            [
                e_spatial,
                knn_edges,
            ],
            axis=-1,
        )
        return e_spatial
    
    def append_random_pairs(self, e_spatial, query_indices, spatial_len):
        n_random = int(self.hparams["randomisation"] * len(query_indices))
        indices_src = torch.randint(0, len(query_indices), (n_random,), device=self.device)
        indices_dest = torch.randint(0, spatial_len, (n_random,), device=self.device)
        random_pairs = torch.stack([query_indices[indices_src], indices_dest])


        e_spatial = torch.cat(
            [
                e_spatial,
                random_pairs
            ],
            axis=-1,
        )
        return e_spatial
    
    def get_true_pairs(self, e_spatial, y_cluster, new_weights, e_bidir):
        e_spatial = torch.cat(
            [
                e_spatial.to(self.device),
                e_bidir,
            ],
            axis=-1,
        )
        y_cluster = torch.cat([y_cluster.int(), torch.ones(e_bidir.shape[1])])
        new_weights = torch.cat(
            [
                new_weights,
                torch.ones(e_bidir.shape[1], device=self.device)
                * self.hparams["weight"],
            ]
        )
        return e_spatial, y_cluster, new_weights
    
    def get_hinge_distance(self, spatials, e_spatial, y_cluster):
    
        hinge = y_cluster.float().to(self.device)
        hinge[hinge == 0] = -1

        reference = spatials[0].index_select(0, e_spatial[1])
        neighbors = spatials[0].index_select(0, e_spatial[0])
        if self.hparams["normalize"]:
            d = torch.acos((1-1e-3)*self.cos(reference, neighbors))
        else:
            d = (reference - neighbors).square().sum(-1)
            d = torch.sqrt(d + 1e-12)
        
        for i in range(1, len(spatials)):
            reference = spatials[i].index_select(0, e_spatial[1])
            neighbors = spatials[i].index_select(0, e_spatial[0])
            if self.hparams["normalize"]:
                d_2 = torch.acos((1-1e-3)*self.cos(reference, neighbors))
            else:
                d_2 = (reference - neighbors).square().sum(-1)
                d_2 = torch.sqrt(d_2 + 1e-12)
            d = torch.minimum(d,d_2)
        
        return hinge, d

    def get_truth(self, batch, e_spatial, e_bidir, pid):
        
        e_spatial_easy_fake = e_spatial[:, pid[e_spatial[0]] != pid[e_spatial[1]]]
        y_cluster_easy_fake = torch.zeros(e_spatial_easy_fake.shape[1])
        
        e_spatial_ambiguous = e_spatial[:, pid[e_spatial[0]] == pid[e_spatial[1]]]
        e_spatial_ambiguous, y_cluster_ambiguous = graph_intersection(e_spatial_ambiguous, e_bidir)
        
        e_spatial = torch.cat([e_spatial_easy_fake.cpu(), e_spatial_ambiguous], dim=-1)
        y_cluster = torch.cat([y_cluster_easy_fake, y_cluster_ambiguous])
        
        return e_spatial, y_cluster
    
    def training_step(self, batch, batch_idx):

        """
        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """
            
        # Instantiate empty prediction edge list
        e_spatial = torch.empty([2, 0], dtype=torch.int64, device=self.device)
        
        # Forward pass of model, handling whether Cell Information (ci) is included
        input_data = self.get_input_data(batch, data_type = "signal_")       
        
        with torch.no_grad():
            spatials = self(input_data)
            if self.hparams["normalize"]:
                spatials = nn.functional.normalize(spatials, p=2.0, dim=2, eps=1e-12)

        query_indices, queries = self.get_query_points(batch, spatials)

        # Append Hard Negative Mining (hnm) with KNN graph
        
        if "hnm" in self.hparams["regime"]:
            e_spatial = self.multi_append_hnm_pairs(e_spatial, queries, query_indices, spatials)
        
        # Append random edges pairs (rp) for stability
        if "rp" in self.hparams["regime"]:
            e_spatial = self.append_random_pairs(e_spatial, query_indices, len(spatials[0]))
            
        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        e_bidir = torch.cat(
            [batch["signal_true_edges"], batch["signal_true_edges"].flip(0)], axis=-1
        )
        
        # Calculate truth from intersection between Prediction graph and Truth graph
        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir, batch["signal_pid"])
        new_weights = y_cluster.to(self.device) * self.hparams["weight"]

        # Append all positive examples and their truth and weighting
        e_spatial, y_cluster, new_weights = self.get_true_pairs(e_spatial, y_cluster, new_weights, e_bidir)
        
        included_hits = e_spatial.unique()        
        spatials[:,included_hits] = self(input_data[included_hits])
        if self.hparams["normalize"]:
            spatials = nn.functional.normalize(spatials, p=2.0, dim=2, eps=1e-12)

        cosine_hinge, cosine_d = self.get_hinge_distance(spatials, e_spatial, y_cluster)

        # Give negative examples a weight of 1 (note that there may still be TRUE examples that are weightless)
        new_weights[
            y_cluster == 0
        ] = 1  
        # d = cosine_d * new_weights

        cosine_loss = new_weights * torch.nn.functional.hinge_embedding_loss(
            cosine_d, cosine_hinge, margin=self.hparams["margin"], reduction="none"
        )
        cosine_loss = cosine_loss.square().mean()

        self.log("train_loss", cosine_loss)

        return cosine_loss

    def shared_evaluation(self, batch, batch_idx, knn_radius, knn_num, log=False, stage = "val"):
        
        if self.hparams["val_on_noise"]:
            data_type = ""
            truth_definition = "modulewise_true_edges"
            pid = "pid"
        else:
            data_type = "signal_"
            truth_definition = "signal_true_edges"
            pid = "signal_pid"

        input_data = self.get_input_data(batch, data_type = data_type)    
        spatials = self(input_data)
        if self.hparams["normalize"]:
            spatials = nn.functional.normalize(spatials, p=2.0, dim=2, eps=1e-12)

        e_bidir = torch.cat(
            [batch[truth_definition],
             batch[truth_definition].flip(0)], axis=-1
        )
                
        _, truth_distance = self.get_hinge_distance(spatials, batch[truth_definition].to(self.device), torch.ones(batch[truth_definition].shape[1]))
        truth_distance, _ = torch.sort(truth_distance)
        truth_distance_eff99 = truth_distance[int(0.99*len(truth_distance))]
        truth_distance_eff95 = truth_distance[int(0.95*len(truth_distance))]
        
        knn_radius = min(knn_radius, truth_distance_eff99.item())
        
        # Build whole KNN graph
        e_spatial = multi_build_edges(spatials, spatials, indices=None, r_max=knn_radius, k_max=knn_num, stage = stage)
        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir, batch[pid])

        cluster_true = e_bidir.shape[1]
        cluster_true_positive = y_cluster.sum()
        cluster_positive = e_spatial.shape[1]

        eff = torch.tensor(cluster_true_positive / cluster_true)
        pur = torch.tensor(cluster_true_positive / cluster_positive)

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {"truth_distance_eff99": truth_distance_eff99, "truth_distance_eff95": truth_distance_eff95, "eff": eff, "pur": pur, "current_lr": current_lr}
            )
        logging.info("Efficiency: {}".format(eff))
        logging.info("Purity: {}".format(pur))
        logging.info(batch.event_file)
        
        return {
            "truth_distance_eff99": truth_distance_eff99,
            "truth_distance_eff95": truth_distance_eff95,
            "preds": e_spatial,
            "truth": y_cluster,
            "truth_graph": e_bidir,
            "eff": eff,
            "pur": pur,
        }
    
    def validation_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """

        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_val"], 500, log=True
        )

        return outputs
    

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_test"], 1000, log=False, stage = "test"
        )
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