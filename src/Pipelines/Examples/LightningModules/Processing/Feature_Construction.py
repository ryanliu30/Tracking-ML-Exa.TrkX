# System imports
import sys
import os
import multiprocessing as mp
from functools import partial

# 3rd party imports
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.nn import Linear
import torch.nn as nn

# Local imports
from .utils import prepare_event


class Feature_Store(LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.input_dir = self.hparams['input_dir']
        self.output_dir = self.hparams['output_dir']
        self.n_files = self.hparams['n_files']

        self.n_tasks = self.hparams['n_tasks']
        self.task = 0 if "task" not in self.hparams else self.hparams['task']
        self.n_workers = self.hparams['n_workers']

    def prepare_data(self):
        # Find the input files
        all_files = os.listdir(self.input_dir)
        all_events = sorted(np.unique([os.path.join(self.input_dir, event[:14]) for event in all_files]))[:self.n_files]

        # Split the input files by number of tasks and select my chunk only
        all_events = np.array_split(all_events, self.n_tasks)[self.task]

        # Prepare output
        # output_dir = os.path.expandvars(self.output_dir) FIGURE OUT HOW TO USE THIS!
        os.makedirs(self.output_dir, exist_ok=True)
        print('Writing outputs to ' + self.output_dir)

        # Process input files with a worker pool
        with mp.Pool(processes=self.n_workers) as pool:
            process_func = partial(prepare_event, **self.hparams)
            pool.map(process_func, all_events)