#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install einops')
get_ipython().system('pip install lightning')
get_ipython().system('pip install wandb')
get_ipython().system('pip install -qqq wandb pytorch-lightning')


# In[ ]:


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
from torch.utils.data import DataLoader, TensorDataset, random_split
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import StepLR
from lightning.pytorch.callbacks import ModelCheckpoint

wandb.login()

