import sys
import os
import argparse
import yaml
import time

import torch
import numpy
import random
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append('../..')

from LightningModules.TrackML_ACAT.training_utils import model_selector

import wandb

from pytorch_lightning.plugins import DDPPlugin, DDP2Plugin, DDPSpawnPlugin
from pytorch_lightning.overrides import LightningDistributedModule

from pytorch_lightning import seed_everything


class CustomDDPPlugin(DDPPlugin):
    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = self._setup_model(LightningDistributedModule(self.model))
        self._register_ddp_hooks()
        self._model._set_static_graph()


def set_random_seed(seed):
    torch.random.manual_seed(seed)
    print("Random seed:", seed)
    seed_everything(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("train_gnn.py")
    add_arg = parser.add_argument
    add_arg("model", nargs="?", default=None)
    add_arg("config", nargs="?", default="default_config.yaml")
    add_arg("checkpoint", nargs="?", default=None)
    add_arg("random_seed", nargs="?", default=None)
    return parser.parse_args()


def main():
    print("Running main")
    print(time.ctime())

    args = parse_args()

    with open(args.config) as file:
        print("Using config file: {}".format(args.config))
        default_configs = yaml.load(file, Loader=yaml.FullLoader)

    if args.checkpoint is not None:
        default_configs = torch.load(args.checkpoint)["hyper_parameters"]
    
    # Set random seed
    if args.random_seed is not None:
        set_random_seed(args.random_seed)
        default_configs["random_seed"] = args.random_seed

    elif "random_seed" in default_configs.keys():
        set_random_seed(default_configs["random_seed"])

    print("Initialising model")
    print(time.ctime())
    if "SLURM_JOB_ID" in os.environ:
        default_root_dir = os.path.join("/global/cfs/cdirs/m3443/usr/ryanliu/TrackML/", os.environ["SLURM_JOB_ID"])
    else: 
        default_root_dir = "/global/cfs/cdirs/m3443/usr/ryanliu/TrackML/"

    model = model_selector(str(args.model), default_configs)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=2, save_last=True
    )
    
    logger = WandbLogger(
        project=default_configs["project"],
        group=os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else None
    )
    
    # logger.watch(model, log="all")
    if os.environ["SLURM_PROCID"] == '0':
        print("SLURM_PROCID found")
        os.makedirs(default_root_dir, exist_ok=True)
        
        
    trainer = Trainer(
        gpus=default_configs["gpus"],
        num_nodes=default_configs["nodes"],
        max_epochs=model.hparams["max_epochs"],
        logger=logger,
        strategy=CustomDDPPlugin(find_unused_parameters=False),
        callbacks=[checkpoint_callback],
        default_root_dir=default_root_dir
    )
    trainer.fit(model, ckpt_path=args.checkpoint)


if __name__ == "__main__":

    main()