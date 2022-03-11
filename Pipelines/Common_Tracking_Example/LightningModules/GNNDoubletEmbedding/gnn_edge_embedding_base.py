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
from .utils import load_dataset_paths, EdgeEmbeddingDataset, find_neighbors, build_neighbors_list

class EdgeEmbeddingBase(LightningModule):
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
        self.trainset = EdgeEmbeddingDataset(self.trainset, self.hparams, stage = "train", device = "cpu")
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=1)
        else:
            return None

    def val_dataloader(self):
        self.valset = EdgeEmbeddingDataset(self.valset, self.hparams, stage = "val", device = "cpu")
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=1)
        else:
            return None

    def test_dataloader(self):
        self.testset = EdgeEmbeddingDataset(self.testset, self.hparams, stage = "test", device = "cpu")
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
    
    def get_true_triplets(self, batch, ind, neighbors):

        # Build Triplets by constructing fully connected graph at each node 
        lengths = (neighbors >= 0).sum(-1)
        triplets = torch.arange(neighbors.shape[1], device = self.device).repeat((neighbors.shape[0], neighbors.shape[1], 1))
        
        # cut the triplets consists of non-existing doublets in third axis
        # to be honest this might not be needed since existence mask can do its work anyway
        triplets[triplets >= lengths.reshape((-1, 1, 1))] = -1
        # removing self-loops
        triplets[triplets == torch.arange(neighbors.shape[1], device = self.device).unsqueeze(0).unsqueeze(2).expand(neighbors.shape[0], -1, neighbors.shape[1])] = -1
        
        # return the mask of non-existing doublets for further usage
        existence_mask = neighbors >= 0

        # Get weights
        weights = self.pt_to_weight(batch.pt[ind])
        # balance nodes with high connectivity (might have a more natural way to do?)
        weights = weights/((neighbors >= 0).sum(-1) + 1e-12)

        # Weighting merged nodes with higher weights to compensate the punishment endured in fake section
        merged_nodes = (batch.pid[ind.unsqueeze(1).expand(neighbors.shape)] != batch.pid[neighbors]).any(1)
        weights[merged_nodes] = self.hparams["merged_weight_ratio"] * weights[merged_nodes]

        weights = weights.reshape((-1, 1, 1)).expand(triplets.shape)
        
        return triplets, weights, existence_mask
    
    def get_fake_triplets(self, embedding1, embedding2, batch, ind, idxs, r_max, k_max):
        # Get triplets with FRNN search
        with torch.no_grad():
            triplets = find_neighbors(embedding1, embedding2, idxs, r_max=r_max, k_max=k_max)
        indices = ind.repeat((idxs.shape[1], 1)).T.long()

        # Filering out PID fake & not existing idxs
        existence_mask = (idxs >= 0)
        pid_mask = batch.pid[indices.long()] != batch.pid[idxs.long()]

        # Get weights for each node
        weights = self.pt_to_weight(batch.pt[ind])
        weights = weights/((idxs >= 0).sum(-1) + 1e-12)
        weights = weights.reshape((-1, 1, 1)).expand(triplets.shape)
        
        return triplets, weights, existence_mask, pid_mask
        
    def get_hinge_distance(self, embedding1, embedding2, batch, r_max = 1.0, k_max = 10, mode = "pid_fake"):
        
        if mode == "pid_fake":
            triplets, weights, existence_mask, pid_mask = self.get_fake_triplets(embedding1, embedding2, batch, batch.ind, batch.idxs, r_max, k_max)
            
        elif mode == "modulewise_truth":
            neighbors = build_neighbors_list(batch.pid, batch.modulewise_true_edges, device = self.device)[batch.ind].long()
            triplets, weights, existence_mask= self.get_true_triplets(batch, batch.ind, neighbors)
            pid_mask = None
            
        else: raise Exception("Please input a valid mode")
        
        # Since double index select is still not available for torch, flatten doublet embedding to get hinge distance
        
        # (i, j, k) = i*knn + j
        in_edges = torch.arange(triplets.shape[0]*triplets.shape[1],device = self.device)
        in_edges = in_edges.reshape((triplets.shape[0], triplets.shape[1])).unsqueeze(2).expand(-1,-1, triplets.shape[2])
        # (i, j, k) = i*knn + Tri(i,j,k)
        out_edges = (triplets + (torch.arange(triplets.shape[0], device = self.device) * triplets.shape[1]).reshape((-1, 1, 1)))
        
        # flatten them according to triplets positions
        in_edges = in_edges[triplets >= 0]
        out_edges = out_edges[triplets >= 0]
        weights = weights[triplets >= 0]
        
        # flatten the first two dimensions of embeddings (i.e. the shape of adjacency list)
        embedding1 = embedding1.reshape((-1, embedding1.shape[-1]))
        embedding2 = embedding2.reshape((-1, embedding2.shape[-1]))
        
        existence_mask = existence_mask.flatten()
        
        # mask out any triplet involving not existing doublets
        mask = existence_mask[in_edges] & existence_mask[out_edges]
        if pid_mask is not None:
            
            # for fake section we also need to mask out any pid true triplet
            pid_mask = pid_mask.flatten()
            mask = mask & (pid_mask[in_edges] | pid_mask[out_edges])
        
        # hinge distance given by geodesic distance
        d = torch.acos((1-1e-4)*self.cos(embedding1[in_edges], embedding2[out_edges]))

        d = d[mask]
        weights = weights[mask]
        
        # might not needed, just to avoid any nan 
        weights = weights[d == d]
        d = d[d == d]
        weights = weights/(weights.sum() + 1e-12)
        
        if mode == "pid_fake":
            hinge = - torch.ones(len(d), device = self.device)
        elif mode == "modulewise_truth":
            hinge = torch.ones(len(d), device = self.device)
        
        return d, hinge, weights

    
    def training_step(self, batch, batch_idx):
        
        input_data = self.get_input_data(batch)
        
        # inference section
        # get fake samples from output of upstream pipeline
        fake_positive_idxs = batch.idxs >= 0
        fake_edges = torch.stack([batch.ind.unsqueeze(1).expand(batch.idxs.shape)[fake_positive_idxs], batch.idxs[fake_positive_idxs]]).long()
        
        # get true samples from modulewise truth, selecting only the doublets containing the nodes in ind
        neighbors = build_neighbors_list(batch.pid, batch.modulewise_true_edges, device = self.device)[batch.ind]
        true_positive_idxs = neighbors >= 0
        true_edges = torch.stack([batch.ind.unsqueeze(1).expand(neighbors.shape)[true_positive_idxs], neighbors[true_positive_idxs]]).long()
        
        # forward propagate them together
        output1, output2 = self(
            input_data,
            torch.cat([fake_edges, true_edges], dim = 1)
        )
        
        # Fake section 
        # Formatting the embeddings in (node smaples, maximum connectivity, embedding dimension)
        fake_embedding1 = torch.zeros(batch.idxs.shape[0], batch.idxs.shape[1], self.hparams["emb_dim"], device = self.device)
        fake_embedding1[fake_positive_idxs] = output1[:fake_edges.shape[1],:]
        
        fake_embedding2 = torch.zeros(batch.idxs.shape[0], batch.idxs.shape[1], self.hparams["emb_dim"], device = self.device)
        fake_embedding2[fake_positive_idxs] = output2[:fake_edges.shape[1],:]
        
        fake_d, fake_hinge, fake_weights = self.get_hinge_distance(fake_embedding1,
                                                                   fake_embedding2,
                                                                   batch,
                                                                   r_max = self.hparams["r_train"],
                                                                   k_max = self.hparams["knn"],
                                                                   mode = "pid_fake")
        
        # True section
        # pretty similar to fake section
        true_embedding1 = torch.zeros(neighbors.shape[0], neighbors.shape[1], self.hparams["emb_dim"], device = self.device)
        true_embedding1[true_positive_idxs] = output1[fake_edges.shape[1]:,:]
        
        true_embedding2 = torch.zeros(neighbors.shape[0], neighbors.shape[1], self.hparams["emb_dim"], device = self.device)
        true_embedding2[true_positive_idxs] = output2[fake_edges.shape[1]:,:]
        
        true_d, true_hinge, true_weights = self.get_hinge_distance(true_embedding1,
                                                                   true_embedding2,
                                                                   batch,
                                                                   mode = "modulewise_truth"
                                                                  )
        
        d = torch.cat([fake_d, true_d], dim = 0)
        hinge = torch.cat([fake_hinge, true_hinge], dim = 0)
        # normalize the weights so that they sum to one
        weights = torch.cat([fake_weights, self.hparams["weight_ratio"]*true_weights], dim = 0)/(1 + self.hparams["weight_ratio"])
        
        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"], reduction="none"
        )

        loss = torch.dot(weights.to(loss), loss.square())/self.hparams["margin"]**2
            
        self.log("train_loss", loss)

        return loss
    
    def get_truth_val_triplets(self, batch, full_ind, sample_edges, input_data):
        full_d = []
        neighbors_list = build_neighbors_list(batch.pid, sample_edges, device = self.device)
        # chunk truth graph into three chunks to avoid memory issues
        for ind in torch.chunk(full_ind, 3, dim = 0):
            neighbors = neighbors_list[ind].long()
            positive_idxs = neighbors >= 0
            edges = torch.stack([ind.unsqueeze(1).expand(neighbors.shape)[positive_idxs], neighbors[positive_idxs]]).long()

            true_output1, true_output2 = self(
                input_data,
                edges
            )

            true_embedding1 = torch.zeros(neighbors.shape[0], neighbors.shape[1], self.hparams["emb_dim"], device = self.device)
            true_embedding1[positive_idxs] = true_output1
            
            true_embedding2 = torch.zeros(neighbors.shape[0], neighbors.shape[1], self.hparams["emb_dim"], device = self.device)
            true_embedding2[positive_idxs] = true_output2

            triplets, _, mask = self.get_true_triplets(batch,
                                                       ind,
                                                       neighbors
                                                      )
            # (i, j, k) = i*knn + j
            in_edges = torch.arange(triplets.shape[0]*triplets.shape[1],device = self.device)
            in_edges = in_edges.reshape((triplets.shape[0], triplets.shape[1])).unsqueeze(2).expand(-1,-1, triplets.shape[2])
            # (i, j, k) = i*knn + Tri(i,j,k)
            out_edges = (triplets + (torch.arange(triplets.shape[0], device = self.device) * triplets.shape[1]).reshape((-1, 1, 1)))

            in_edges = in_edges[triplets >= 0]
            out_edges = out_edges[triplets >= 0]

            # Masking undesired edges
            mask = mask.flatten()
            mask = mask[in_edges] & mask[out_edges]

            true_embedding1 = true_embedding1.reshape((-1, true_embedding1.shape[2]))
            true_embedding2 = true_embedding2.reshape((-1, true_embedding2.shape[2]))
            
            # Instead of geodesic distance, use euclidean distance which is compatible with FRNN
            d = (true_embedding1[in_edges] - true_embedding2[out_edges]).square().sum(-1).sqrt()
                
            full_d.append(d[mask].clone().detach())
            
        return torch.cat(full_d, dim = 0)
    
    def shared_evaluation(self, batch, batch_idx, log=False):

        """
        This method is shared between validation steps and test steps
        """
        
        input_data = self.get_input_data(batch)
        full_ind = torch.arange(batch.pid.shape[0], device = self.device)
        
        d_found = self.get_truth_val_triplets(batch, full_ind, batch.found_edges, input_data)
        d_found, _ = d_found.sort()
        
        # find the radius that can satify the efficiency requirement on found subset with pt cut
        radius = d_found[int(self.hparams["max_eff"]*len(d_found))].clone().detach().item()
        truth_true_positive = (d_found <= radius).sum()
        
        # For purity purpose we also need the number of positive triplets in found subset without cut
        d_all = self.get_truth_val_triplets(batch, full_ind, batch.found_edges_nocut, input_data)
        truth_true_positive_all = (d_all <= radius).sum()
        
        # For denominator of efficiency, we can simply calculate it with math: n(n-1)
        truth_true = (build_neighbors_list(batch.pid, batch.signal_edges, device = self.device) >= 0).sum(-1)
        truth_true = (truth_true * (truth_true - 1)).sum()
        
        # Validate All Samples
        
        prediction_positive = torch.tensor([0], device = self.device)
        prediction_positive_cut = torch.tensor([0], device = self.device)
        
        # unresolved: i dont know why the chucks get an extra index so i have to use idxs[0] instead of just idxs
        for (idxs, ind) in zip(batch.idxs[0], batch.ind[0]):
            positive_idxs = idxs >= 0
            edges = torch.stack([ind.unsqueeze(1).expand(idxs.shape)[positive_idxs], idxs[positive_idxs]]).long()
            
            output1, output2 = self(
                input_data,
                edges
            )
            mask = torch.zeros(idxs.shape, device = self.device).bool()
            
            # mask out low pt tracks for cut metrics
            mask[positive_idxs] = ((batch.pt[idxs[positive_idxs]] > self.hparams["signal_pt_cut"]) |
                                   (batch.pt[ind.unsqueeze(1).expand(idxs.shape)[positive_idxs]] > self.hparams["signal_pt_cut"]))

            embedding1 = torch.zeros(idxs.shape[0], idxs.shape[1], self.hparams["emb_dim"], device = self.device)
            embedding1[positive_idxs] = output1
            embedding2 = torch.zeros(idxs.shape[0], idxs.shape[1], self.hparams["emb_dim"], device = self.device)
            embedding2[positive_idxs] = output2
            
            triplets = find_neighbors(embedding1, embedding2, idxs, r_max = radius, k_max = 100)
            # get the graph size
            prediction_positive = prediction_positive + (triplets >= 0).sum()
            prediction_positive_cut = prediction_positive_cut + (triplets[mask] >= 0).sum()
            
        
        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            
            self.log_dict(
                {
                    "dist@{}".format(self.hparams["max_eff"]): radius,
                    "eff": (truth_true_positive/truth_true).clone().detach(),
                    "pur": (truth_true_positive_all/prediction_positive).clone().detach(),
                    "cut_pur": (truth_true_positive/prediction_positive_cut).clone().detach(),
                    "current_lr": current_lr,
                }
            )
        return {
                "dist@{}".format(self.hparams["max_eff"]): radius,
                "eff": (truth_true_positive/truth_true).clone().detach(),
                "pur": (truth_true_positive_all/prediction_positive).clone().detach(),
                "cut_pur": (truth_true_positive/prediction_positive_cut).clone().detach(),
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