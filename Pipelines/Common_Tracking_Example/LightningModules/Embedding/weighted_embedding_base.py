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
from .utils import graph_intersection, split_full_datasets, multi_build_edges

device = "cuda" if torch.cuda.is_available() else "cpu"


class WeightedEmbeddingBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)
        self.cos = nn.CosineSimilarity()
        

    def setup(self, stage):
        self.trainset, self.valset, self.testset = split_full_datasets(
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

    def get_input_data(self, batch, data_type = "signal"):
        
        if data_type == "signal":
            ci = "signal_cell_data"
            x = "signal_x"
        else:
            ci = "cell_data"
            x = "x"
        
        if "ci" in self.hparams["regime"]:
            input_data = torch.cat([batch[ci][:, :self.hparams["cell_channels"]], batch[x]], axis=-1)
            
        else:
            input_data = batch[x]
            
        return input_data
    
    def get_query_points(self, batch, spatials):
        
        query_indices = batch.signal_true_edges.unique()
        query_indices = query_indices[torch.randperm(len(query_indices))][:self.hparams["signal_points_per_batch"]]
        
        queries = []
        for spatial in spatials:
            queries.append(spatial[query_indices])
        
        return query_indices, queries
    
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
    
    def get_true_pairs(self, e_spatial, y_cluster, e_bidir):
        e_spatial = torch.cat(
            [
                e_spatial.to(self.device),
                e_bidir,
            ],
            axis=-1,
        )
        y_cluster = torch.cat([y_cluster.int(), torch.ones(e_bidir.shape[1])])
        
        return e_spatial, y_cluster
    
    def get_hinge_distance(self, spatials, e_spatial, y_cluster):
    
        hinge = y_cluster.float().to(self.device)
        e_spatial = e_spatial.to(self.device)
        hinge[hinge == 0] = -1

        reference = spatials[0].index_select(0, e_spatial[1])
        neighbors = spatials[0].index_select(0, e_spatial[0])
        if self.hparams["normalize"]:
            d = torch.acos((1-1e-4)*self.cos(reference, neighbors))
        else:
            d = (reference - neighbors).square().sum(-1)
            d = torch.sqrt(d + 1e-12)
        
        for i in range(1, len(spatials)):
            reference = spatials[i].index_select(0, e_spatial[1])
            neighbors = spatials[i].index_select(0, e_spatial[0])
            if self.hparams["normalize"]:
                d_2 = torch.acos((1-1e-4)*self.cos(reference, neighbors))
            else:
                d_2 = (reference - neighbors).square().sum(-1)
                d_2 = torch.sqrt(d_2 + 1e-12)
            d = torch.minimum(d,d_2)
        
        return hinge, d

    def get_truth(self, e_spatial, e_bidir, pid, punish_same_pid):
        
        e_spatial_easy_fake = e_spatial[:, pid[e_spatial[0]] != pid[e_spatial[1]]]
        y_cluster_easy_fake = torch.zeros(e_spatial_easy_fake.shape[1])
        
        if not punish_same_pid:
            return e_spatial_easy_fake.cpu(), y_cluster_easy_fake
        
        else:
            e_spatial_ambiguous = e_spatial[:, pid[e_spatial[0]] == pid[e_spatial[1]]]
            e_spatial_ambiguous, y_cluster_ambiguous = graph_intersection(e_spatial_ambiguous, e_bidir)

            e_spatial = torch.cat([e_spatial_easy_fake.cpu(), e_spatial_ambiguous], dim=-1)
            y_cluster = torch.cat([y_cluster_easy_fake, y_cluster_ambiguous])

            return e_spatial, y_cluster
        
    def pt_to_weight(self, pt):
        
        pt[pt!=pt] = 0

        h = lambda i: torch.heaviside(i, torch.zeros(1).to(pt))
        minimum = lambda i: torch.minimum(i, torch.ones(1).to(pt))
        
        eps = self.hparams["weight_leak"]
        cut = self.hparams["signal_pt_cut"] - self.hparams["signal_pt_interval"]
        cap = self.hparams["signal_pt_cut"]
        
        return minimum(h(pt-cut)*(pt-cut)/(cap-cut)) + (eps * h(pt-cap) * (pt-cap))
    
    def get_edge_weight(self, e_spatial, y_cluster, pt):
        
        weight = self.pt_to_weight(pt[e_spatial[0]]) + self.pt_to_weight(pt[e_spatial[1]])
        unnormalized_weight = weight.clone().detach()
        
        truth_weight = weight[y_cluster != 0].sum()
        fake_weight = weight[y_cluster == 0].sum()
        
        if (y_cluster != 0).any(0):
            weight[y_cluster != 0] = self.hparams["weight_ratio"] * weight[y_cluster != 0] / truth_weight
        if (y_cluster == 0).any(0):
            weight[y_cluster == 0] = weight[y_cluster == 0] / fake_weight
        
        return weight/(1+self.hparams["weight_ratio"]), unnormalized_weight
    
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
        input_data = self.get_input_data(batch, data_type = "signal")       
        
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
        e_spatial, y_cluster = self.get_truth(e_spatial, e_bidir, batch["signal_pid"], self.hparams["punish_same_pid"])

        # Append all positive examples and their truth and weighting
        e_spatial, y_cluster = self.get_true_pairs(e_spatial, y_cluster, e_bidir)
        
        weight, _ = self.get_edge_weight(e_spatial, y_cluster, batch["signal_pt"])
        
        included_hits = e_spatial.unique()        
        spatials[:,included_hits] = self(input_data[included_hits])
        if self.hparams["normalize"]:
            spatials = nn.functional.normalize(spatials, p=2.0, dim=2, eps=1e-12)

        hinge, d = self.get_hinge_distance(spatials, e_spatial, y_cluster)

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"], reduction="none"
        )
        if self.hparams["square"]:
            loss = torch.dot(weight.to(loss), loss.square())/self.hparams["margin"]**2
        else:
            loss = torch.dot(weight.to(loss), loss)/self.hparams["margin"]

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx, knn_radius, knn_num, log=False, stage = "val"):
        
        if self.hparams["val_on_noise"]:
            data_type = ""
            truth_definition = "modulewise_true_edges"
            pid = "pid"
            pt = "pt"
        else:
            data_type = "signal"
            truth_definition = "signal_true_edges"
            pid = "signal_pid"
            pt = "signal_pt"

        input_data = self.get_input_data(batch, data_type = data_type)    
        spatials = self(input_data)
        if self.hparams["normalize"]:
            spatials = nn.functional.normalize(spatials, p=2.0, dim=2, eps=1e-12)
        
        e_bidir = torch.cat(
            [batch[truth_definition],
             batch[truth_definition].flip(0)], axis=-1
        )
        
        truth_mask = (batch[pt][e_bidir] > self.hparams["signal_pt_cut"]).all(0)
        
        _, truth_distance = self.get_hinge_distance(spatials, e_bidir.to(self.device), torch.ones(e_bidir.shape[1]))
        truth_distance_full = truth_distance.clone()
        truth_distance = truth_distance[truth_mask]
        truth_distance, _ = torch.sort(truth_distance)
        
        if self.hparams["normalize"]:
            truth_distance = 2*torch.sin(truth_distance/2)
            truth_distance_full = 2*torch.sin(truth_distance_full/2)
            
        truth_distance_eff98 = truth_distance[int(self.hparams["max_eff"]*len(truth_distance))]

        if self.hparams["max_eff"] < 1:
            knn_radius = min(knn_radius, truth_distance_eff98.item())
        
        # Build whole KNN graph
        e_spatial = multi_build_edges(spatials, spatials, indices=None, r_max=knn_radius, k_max=knn_num, stage = stage)
        e_spatial, y_cluster = self.get_truth(e_spatial, e_bidir, batch[pid], True)
        
        # Get weight
        weight, prediction_weight = self.get_edge_weight(e_spatial, y_cluster, batch[pt])
        _, truth_weight = self.get_edge_weight(e_bidir, torch.ones(e_bidir.shape[1]), batch[pt])
        
        # Get loss
        hinge, d = self.get_hinge_distance(spatials, e_spatial, y_cluster)

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"], reduction="none"
        )
        if self.hparams["square"]:
            loss = torch.dot(weight.to(loss), loss.square())/self.hparams["margin"]**2
        else:
            loss = torch.dot(weight.to(loss), loss)/self.hparams["margin"]
        
        # Get metrics
        prediction_mask = (batch[pt][e_spatial] > self.hparams["signal_pt_cut"]).all(0)

        weighted_cluster_true = truth_weight.sum()
        weighted_cluster_true_positive = prediction_weight[y_cluster != 0].sum()
        weighted_cluster_positive = prediction_weight.sum()
        
        weighted_eff = torch.tensor(weighted_cluster_true_positive / weighted_cluster_true)
        weighted_pur = torch.tensor(weighted_cluster_true_positive / weighted_cluster_positive)
        
        cut_cluster_true = truth_mask.sum()+ 1e-12
        cut_cluster_true_positive = (y_cluster[prediction_mask] != 0).sum()
        cut_cluster_positive = prediction_mask.shape[0] + 1e-12
        cut_eff = torch.tensor(cut_cluster_true_positive / cut_cluster_true)
        cut_pur = torch.tensor(cut_cluster_true_positive / cut_cluster_positive)
        
        cluster_true = e_bidir.shape[1]
        cluster_true_positive = y_cluster.sum()
        cluster_positive = e_spatial.shape[1]

        eff = torch.tensor(cluster_true_positive / cluster_true)
        pur = torch.tensor(cluster_true_positive / cluster_positive)

        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {"val_loss":loss, "r@eff{}".format(self.hparams["max_eff"]): truth_distance_eff98, "weighted_eff":weighted_eff, "weighted_pur":weighted_pur, "cut_eff": cut_eff, "cut_pur": cut_pur, "eff": eff, "pur": pur, "current_lr": current_lr}
            )
        logging.info("Efficiency: {}".format(eff))
        logging.info("Purity: {}".format(pur))
        logging.info(batch.event_file)
        
        return {
            "val_loss":loss,
            "truth_distance":truth_distance_full,
            "truth_weight": truth_weight,
            "weighted_eff":weighted_eff,
            "weighted_pur":weighted_pur,
            "cut_eff": cut_eff,
            "cut_pur": cut_pur,
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

        return outputs["val_loss"]
    

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