import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# from utils.imports import *

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.datasets as datasets
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import StepLR
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
import random

USE_WANDB = False

if USE_WANDB:
    wandb.login()

from models.StandAloneFFN import FFN
from models.StandAloneHyena import HyenaOperator
from data.datagenerator import data

### FunctionClassifier

class FFNClassifier(pl.LightningModule):
    def __init__(
            self, 
            FFN,
            x_data,
            out_data
        ):
        super(FFNClassifier, self).__init__()
        self.FFN = FFN
        self._x_data = x_data
        self._out_data = out_data

    def forward(self,x):
        y = x
        z = rearrange(y, 'b l d -> b (l d)')
        out = self.FFN(z)
        return out

    def prepare_data(self):
        total_dataset = TensorDataset(self._x_data, self._out_data)
        train_size = int(0.8 * len(total_dataset))
        test_size = len(total_dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(total_dataset, [train_size, test_size])

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=20, shuffle=True)
        return train_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=20, shuffle=False)
        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00005)
        scheduler = StepLR(optimizer, step_size=1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)

        # Compute training accuracy
        _, predicted_labels = torch.max(output, dim=0)
        correct = (predicted_labels == target).sum().item()
        accuracy = correct / target.size(0)

        # Log training accuracy
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger = True)

        loss = F.cross_entropy(output, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # return loss
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Compute predicted labels
        _, predicted_labels = torch.max(y_hat, dim=1)
        _, yn = torch.max(y, dim=1)

        # Compute accuracy
        correct = (predicted_labels == yn).sum().item()
        total = y.size(0)
        accuracy = correct / total
        Array_accuracy.append(predicted_labels)
        Actual_labels.append(yn)

        # Log accuracy
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)

        return {'test_accuracy': accuracy}

class HyenaClassifier(pl.LightningModule):
    def __init__(
            self, 
            HyenaOperator,
            FFN,
            x_data,
            out_data
        ):
        super(HyenaClassifier, self).__init__()
        self.Hyena = HyenaOperator
        self.FFN = FFN
        self._x_data = x_data
        self._out_data = out_data

    def forward(self,x):
        y = self.Hyena(x)
        # y = x
        z = rearrange(y, 'b l d -> b (l d)')
        out = self.FFN(z)
        return out

    def prepare_data(self):
        total_dataset = TensorDataset(self._x_data, self._out_data)
        train_size = int(0.8 * len(total_dataset))
        test_size = len(total_dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(total_dataset, [train_size, test_size])

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=20, shuffle=True)
        return train_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=20, shuffle=False)
        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00005)
        scheduler = StepLR(optimizer, step_size=1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)

        # Compute training accuracy
        _, predicted_labels = torch.max(output, dim=0)
        correct = (predicted_labels == target).sum().item()
        accuracy = correct / target.size(0)

        # Log training accuracy
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger = True)

        loss = F.cross_entropy(output, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # return loss
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Compute predicted labels
        _, predicted_labels = torch.max(y_hat, dim=1)
        _, yn = torch.max(y, dim=1)

        # Compute accuracy
        correct = (predicted_labels == yn).sum().item()
        total = y.size(0)
        accuracy = correct / total
        Array_accuracy.append(predicted_labels)
        Actual_labels.append(yn)

        # Log accuracy
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)

        return {'test_accuracy': accuracy}

device = torch.device("cuda")

# Convert data into tensors
my_x_datax = torch.tensor([[[a, b] for a, b in zip(idata[0], idata[1])] for idata in data])
my_x_data = torch.tensor(my_x_datax, dtype=torch.float32)

# print(my_x_data.shape) # Batch x sequence_length x d
labels = torch.tensor([label for _, _, label in data])

Array_accuracy = []
Actual_labels = []

# One-hot encode the labels
my_out_data = torch.zeros(labels.size(0), 3)
my_out_data[torch.arange(labels.size(0)), labels] = 1
my_out_data = torch.tensor(my_out_data, dtype=torch.float32)

# Assigning Values
d_model = 2
batch_size = 1000
sequence_length = 100

train_ratio = 0.8

# Initialize HyenaOperator model
ffn = FFN(layer_sizes=[200, 128, 64, 32, 3])

Hyena = HyenaOperator(d_model=d_model, l_max=sequence_length, order=2, dropout=0.0, filter_dropout=0.0)
hyena_model = HyenaClassifier(Hyena, ffn, my_x_data, my_out_data)

ffn = FFN(layer_sizes=[200, 128, 64, 64, 3])
ffn_model = FFNClassifier(ffn, my_x_data, my_out_data)

checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')

# initiate wandb logger
if USE_WANDB:
    wandb_logger = WandbLogger(project = 'lightning_model',log_model="all")
else:
    wandb_logger = None

# Initialize Trainer and start training
trainer = pl.Trainer(
    accelerator="cpu",
    max_epochs=50,
    logger = wandb_logger,
    log_every_n_steps=1,
)

if USE_WANDB:
    wandb_logger.watch(hyena_model, log="gradients", log_freq=50, log_graph=True)

trainer.fit(ffn_model)
trainer.test()

if USE_WANDB:
    wandb.finish()
