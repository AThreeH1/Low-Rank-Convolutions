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


wandb.login()

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
    device = torch.device("cpu")
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

"""## CNN"""


def run_cnn():
    # Hyperparameters
    batch_size = 32
    num_epochs = 10

    device = torch.device("cpu")
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

class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float=1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None] # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]

class ExponentialModulation(OptimModule):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        modulate: bool=True,
        shift: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x

class HyenaFilter(OptimModule):
    def __init__(
            self,
            d_model,
            emb_dim=3, # dim of input to MLP, augments with positional encoding
            order=16, # width of the implicit MLP
            fused_fft_conv=False,
            seq_len=1024,
            lr=1e-3,
            lr_pos_emb=1e-5,
            dropout=0.0,
            w=1, # frequency of periodic activations
            wd=0, # weight decay of kernel parameters
            bias=True,
            num_inner_mlps=2,
            normalized=False,
            **kwargs
        ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
        """
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = seq_len

        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))

        self.modulation = ExponentialModulation(d_model, **kwargs)

        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)


    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None: k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k

        y = fftconv(x, k, bias)
        return y

# Adding FFN

class HyenaOperator(pl.LightningModule):
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
            nn.Linear(d_model * l_max, ff_hidden_dim),
            nn.ReLU(),
            *[nn.Linear(ff_hidden_dim, ff_hidden_dim) for _ in range(ff_num_layers - 1)],
            nn.Linear(ff_hidden_dim, 64),
            # nn.ReLU(),
            # nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.save_hyperparameters()

    def forward(self, u, *args, **kwargs):
        l = u.size(-2)
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
device = torch.device("cpu")
d_model = 100
batch_size = 1000
sequence_length = 2
train_ratio = 0.8


# Initialize HyenaOperator model
hyena_model = HyenaOperator(d_model=d_model, l_max=sequence_length, order=8, dropout=0.0, filter_dropout=0.0)

checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')

# initiate wandb logger
wandb_logger = WandbLogger(project = 'lightning_model',log_model="all")

# Initialize Trainer and start training
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=10,
    logger = wandb_logger,
    log_every_n_steps=1,
)

wandb_logger.watch(hyena_model, log="gradients", log_freq=50, log_graph=True)

trainer.fit(hyena_model)
trainer.test()

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