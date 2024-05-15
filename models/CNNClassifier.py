import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Append the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utils.imports import *
from data.datagenerator import data

USE_WANDB = False
if USE_WANDB:
    wandb.login(key = ['7e169996e30d15f253a5f1d92ef090f2f3b948c4'])

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
        total_dataset = TensorDataset(x_data, labels)
        train_size = int(0.8 * len(total_dataset))
        test_size = len(total_dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(total_dataset, [train_size, test_size])

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

# Convert data into tensors
x_datax = torch.tensor([[[a, b] for a, b in zip(idata[0], idata[1])] for idata in data])
x_data = x_datax.permute(0, 2, 1)
x_data = torch.tensor(x_data, dtype=torch.float32)
labels = torch.tensor([label for _, _, label in data])
labels = torch.tensor(labels, dtype=torch.long)

Array_accuracy = []
Actual_labels = []

# Initialize model
model = CNNClassifier()

# initiate wandb logger
if USE_WANDB:
    wandb_logger = WandbLogger(project = 'lightning_model',log_model="all")
else:
    wandb_logger = None

# Initialize Lightning Trainer
trainer = pl.Trainer(
    accelerator = "cpu", 
    max_epochs = 11,
    logger= wandb_logger,
    log_every_n_steps=1
)

if USE_WANDB:
    wandb_logger.watch(model, log="gradients", log_freq=50, log_graph=True)

trainer.fit(model)
trainer.test()

if USE_WANDB:
    wandb.finish()