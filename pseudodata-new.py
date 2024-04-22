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
from sklearn.model_selection import train_test_split
import random
import pandas as pd


# wandb.login()

"""## Data"""

data = []

for r in range(1000):

    # initialising dimentions d1 and d2 with random numbers
    d1 = []
    d2 = []

    for i in range(100):
        d1.append((random.random()) - 0.5)
        d2.append((random.random()) - 0.5)

    # initializing length of signal 'l' in dimension d1 & d2
    l = (random.randint(10, len(d1)/2))
    # print('length of main signal = ', l)

    # initializing the point 'm' from where the signal starts
    m = (random.randint(0, (len(d1)-l)))
    # print('The signal starts from index ', m)

    # # Initialising the three signals s1, s2 and s3 in dimension d2
    # print('Signal s1 is triangular wave')
    # print('Signal s2 is square wave')
    # print('Signal s3 is downward semicircular wave')

    # s1 - triangular wave
    def s1(t):
        if t < 0.5:
            return t
        else:
            return 1 - t

    # s2 - square wave
    def s2(t):
        if 0 <= t < 0.5:
            return 0.5
        else:
            return -0.5

    # s3 - lower semicircle
    def s3(t):
        return (-(0.5**2 - (t - 0.5)**2)**0.5)

    # selecting either of s1, s2 or s3 at random
    p = random.randint(1,3)
    # print('Target signal of s', p, ' is selected')

    # Inputting the signals in the respective dimensions
    for i in range(l):

        d1[m] = (np.sin(i*2*np.pi/(l-1)))/2

        j = i/(l-1)
        if p == 1:
            d2[m] = s1(j)
        if p == 2:
            d2[m] = s2(j)
        if p == 3:
            d2[m] = s3(j)

        m += 1

    if len(d2) - m >= 10:

        ps = random.randint(1,3)
        # print('Post signal of s', ps, ' is selected')

        Q = random.randint(10, min((len(d1) - m), 50))
        # print('Post noise is of length = ', Q)

        R = random.randint(0, (len(d1) - m - Q))

        post = m + R
        # print('Post noise starts from index ', post)

        for i in range(Q):
            j = i/(Q-1)
            if ps == 1:
                d2[post] = s1(j)
            if ps == 2:
                d2[post] = s2(j)
            if ps == 3:
                d2[post] = s3(j)

            post += 1

    if (m - l) >= 10:

        bs = random.randint(1,3)
        # print('Pre signal of s', bs, ' is selected')

        Qn = random.randint(10, min((m - l), 50))
        # print('Pre noise is of length = ', Qn)

        Rn = random.randint(0, (m - l - Qn))

        pre = Rn
        # print('Pre noise starts from index ', pre)

        for i in range(Qn):
            j = i/(Qn-1)
            if bs == 1:
                d2[pre] = s1(j)
            if bs == 2:
                d2[pre] = s2(j)
            if bs == 3:
                d2[pre] = s3(j)

            pre += 1

    Batch = (d1, d2, p-1)
    data.append(Batch)

print('data stored')
# => should go into a function


"""## FFN

### Lightning
"""

def run_ffn():
    # Separate inputs and targets
    device = torch.device("cuda")
    inputs = []
    targets = []
    for i in data:
        inputs.append(i[0] + i[1])
        targets.append(i[2])

    # Convert to numpy arrays
    inputs = np.array(inputs)
    targets = np.array(targets)

    # Split data into training and testing sets
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=0.2, random_state=42)

    # Convert numpy arrays to PyTorch tensors
    train_inputs = torch.tensor(train_inputs, dtype=torch.float32).to(device)
    test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(device)
    train_targets = torch.tensor(train_targets, dtype=torch.long).to(device)  # Assuming targets are indices
    test_targets = torch.tensor(test_targets, dtype=torch.long).to(device)


    # Define LightningModule
    class SimpleNN(pl.LightningModule):
        def __init__(self, input_size, num_classes):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_size, 128)
            self.fc2 = torch.nn.Linear(128, 64)
            self.fc3 = torch.nn.Linear(64, 64)
            self.fc4 = torch.nn.Linear(64, num_classes)
            self.train_dataset = None
            self.test_dataset = None

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.softmax(self.fc4(x), dim=1)
            return x

        def prepare_data(self):
            self.train_dataset = TensorDataset(train_inputs, train_targets)
            self.test_dataset = TensorDataset(test_inputs, test_targets)

        def train_dataloader(self):
            train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
            return train_loader

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            preds = torch.argmax(y_hat, dim=1)
            correct = (preds == y).sum().item()
            total = len(y)
            accuracy = correct / total
            self.log('train_loss', loss)
            self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters())

        def test_dataloader(self):
            test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
            return test_loader

        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            test_loss = torch.nn.functional.cross_entropy(y_hat, y)
            preds = torch.argmax(y_hat, dim=1)
            correct = (preds == y).sum().item()
            total = len(y)
            accuracy = correct / total
            self.log("test_loss", test_loss, on_step=False, on_epoch=True)
            self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)
            return {'test_loss': test_loss }

    # Train the model
    input_size = train_inputs.shape[1]
    num_classes = 3  # Assuming 3 classes
    model = SimpleNN(input_size, num_classes)
    trainer = pl.Trainer(accelerator="auto", max_epochs=11)
    trainer.fit(model)

    # Evaluate the model
    trainer.test()

# run_ffn()

"""## CNN"""


def run_cnn():
    # Hyperparameters
    batch_size = 32
    num_epochs = 10

    device = torch.device("cuda")
    inputs = []
    targets = []
    for i in data:
        A = []
        A.append(i[0])
        A.append(i[1])
        inputs.append(A)
        targets.append(i[2])

    # Split data into training and testing sets
    train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=0.2, random_state=42)

    # Convert numpy arrays to PyTorch tensors
    train_inputs = torch.tensor(train_inputs, dtype=torch.float32).to(device)
    test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(device)
    train_targets = torch.tensor(train_targets, dtype=torch.long).to(device)
    test_targets = torch.tensor(test_targets, dtype=torch.long).to(device)

    class CNNClassifier(pl.LightningModule):
        def __init__(self):
            super(CNNClassifier, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3)
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
            self.fc1 = nn.Linear(64 * 23, 128)
            self.fc2 = nn.Linear(128, 3)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool1d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool1d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

        def prepare_data(self):
            self.train_dataset = TensorDataset(train_inputs, train_targets)
            self.test_dataset = TensorDataset(test_inputs, test_targets)

        def train_dataloader(self):
            train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
            return train_loader

        def test_dataloader(self):
            test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
            return test_loader

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            self.log('train_loss', loss)
            return loss

        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            pred = torch.argmax(y_hat, dim=1)
            accuracy = (pred == y).float().mean()
            self.log('test_accuracy', accuracy)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    # Initialize model
    model = CNNClassifier()

    # Initialize Lightning Trainer
    trainer = pl.Trainer(accelerator = "auto", max_epochs = num_epochs + 1)

    # Train the model
    trainer.fit(model) #, train_loader)

    # Test the model
    trainer.test()

# run_cnn()
# xxx

"""## Hyena Hierarchy Lightning"""

def fftconv(u, k, D):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3: k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)

def mul_sum(q, y):
    return (q * y).sum(dim=1)

class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)

class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)

from HyenaLightining import PositionalEmbedding, ExponentialModulation, HyenaFilter

class CosSinHyenaClassifier(pl.LightningModule):
    # TODO
    pass

    def forward(self,x):
        x = hyena(x)
        return fcn(x.flatten())

class HyenaOperator(pl.LightningModule): # Should this not be a OptimModule ?
    def __init__(
            self,
            d_model,
            l_max,
            order=2,
            filter_order=64,
            dropout=0.0,
            filter_dropout=0.0,
            num_classes=3,
            ff_hidden_dim=128,
            ff_num_layers=2,
            **filter_args,
        ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
            num_classes (int): Number of output classes for classification. Defaults to 3.
            ff_hidden_dim (int): Hidden dimension of the feed-forward layers. Defaults to 128.
            ff_num_layers (int): Number of feed-forward layers. Defaults to 2.
        """
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)

        self.short_filter = nn.Conv1d(
            inner_width,
            inner_width,
            3,
            padding=2,
            groups=inner_width
        )
        self.filter_fn = HyenaFilter(
            d_model * (order - 1),
            order=filter_order,
            seq_len=l_max,
            channels=1,
            dropout=filter_dropout,
            **filter_args
        )

        # Feed-forward layers for classification
        self.ff_layers = nn.Sequential(
            # nn.Linear(d_model * l_max, num_classes)
            nn.Linear(d_model * l_max, ff_hidden_dim),
            nn.ReLU(),
            # *[nn.Linear(ff_hidden_dim, ff_hidden_dim) for _ in range(ff_num_layers - 1)],
            # nn.Linear(ff_hidden_dim, ff_hidden_dim),
            nn.Linear(ff_hidden_dim, 64),
            # nn.ReLU(),
            # nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.save_hyperparameters()

    def forward(self, u, *args, **kwargs):
        l = u.size(-2) # This is the ambient dimension of the input !?
        print('l=', l, u.size())
        l_filter = min(l, self.l_max)
        u = self.in_proj(u.float())
        u = rearrange(u, 'b l d -> b d l')

        uc = self.short_filter(u)[...,:l_filter]
        *x, v = uc.split(self.d_model, dim=1)

        k = self.filter_fn.filter(l_filter)[0]
        k = rearrange(k, 'l (o d) -> o d l', o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, '(o d) -> o d', o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], 'b d l -> b l d')

        y = self.out_proj(y)
        y = rearrange(y, 'b l d -> b (l d)')  # Flatten the output
        print('y=', y)
        y = self.ff_layers(y)  # Classification using feed-forward layers
        return y

    def prepare_data(self):
        total_dataset = TensorDataset(x_data, out_data)
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

# Convert data into tensors
x_datax = torch.tensor([[[a, b] for a, b in zip(idata[0], idata[1])] for idata in data])
x_data = x_datax.permute(0, 2, 1)
labels = torch.tensor([label for _, _, label in data])

Array_accuracy = []
Actual_labels = []

# One-hot encode the labels
out_data = torch.zeros(labels.size(0), 3)
out_data[torch.arange(labels.size(0)), labels] = 1

# Assigning Values
device = torch.device("cuda")
d_model = 100
batch_size = 1000
sequence_length = 2
train_ratio = 0.8


# Initialize HyenaOperator model
hyena_model = HyenaOperator(d_model=d_model, l_max=sequence_length, order=2, dropout=0.0, filter_dropout=0.0)

print( x_data[0:20] )
print( hyena_model(x_data[0:20]) )
xxx



checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')

# initiate wandb logger
# wandb_logger = WandbLogger(project = 'lightning_model',log_model="all")

# Initialize Trainer and start training
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=10,
    # logger = wandb_logger,
    log_every_n_steps=1,
)
# wandb_logger.watch(hyena_model, log="gradients", log_freq=50, log_graph=True)

trainer.fit(hyena_model)
trainer.test()

print( hyena_model(x_data[0:20]) )

xxx

wandb.finish()

out_data

concatenated_tensor = torch.cat(Array_accuracy)
Preds = concatenated_tensor.cpu().numpy()
target_concat = torch.cat(Actual_labels)
Target = target_concat.cpu().numpy()
y = np.arange(0,200)
import matplotlib.pyplot as plt

plt.scatter(y,Preds)
# plt.xlabel('Preds')
# plt.ylabel('Target')

# Convert data into tensors
x_datax = torch.tensor([[[a, b] for a, b in zip(idata[0], idata[1])] for idata in data])
x_data = x_datax.permute(0, 2, 1)
labels = torch.tensor([label for _, _, label in data])
print(x_data)

# One-hot encode the labels
out_data = torch.zeros(labels.size(0), 3)
out_data[torch.arange(labels.size(0)), labels] = 1
print(out_data)

# Pass z through the model
with torch.no_grad():
    output = hyena_model(x_data)
    print(output)

# Compute predicted labels
_, predicted_labels = torch.max(output, dim=1)

_, out = torch.max(out_data, dim=1)
print("out = ", out)

# Now, predicted_labels contains the predicted class for each sample in z
print(predicted_labels)

"""## Data visuals"""

import matplotlib.pyplot as plt

x = np.arange(0,100,1)

plt.figure(figsize=(14, 12))

plt.subplot(3, 2, 1)
plt.plot(x,d1)
plt.xlabel('x')
plt.ylabel('d1')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(x,d2)
plt.xlabel('x')
plt.ylabel('d2')
plt.grid(True)

y1 = []
for i in x:
    y1.append(s1(i/99))
plt.subplot(3, 2, 3)
plt.plot(x, y1)
plt.xlabel('x')
plt.ylabel('s1')
plt.grid(True)

y2 = []
for i in x:
    y2.append(s2(i/99))
plt.subplot(3, 2, 4)
plt.plot(x, y2)
plt.xlabel('x')
plt.ylabel('s2')
plt.grid(True)

y3 = []
for i in x:
    y3.append(s3(i/99))
plt.subplot(3, 2, 5)
plt.plot(x, y3)
plt.xlabel('x')
plt.ylabel('s3')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(x, np.sin(x*2*np.pi/99))
plt.xlabel('x')
plt.ylabel('d1_signal')
plt.grid(True)

plt.show()