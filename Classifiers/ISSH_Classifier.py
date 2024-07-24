import os
import sys
import torchinfo
from torchinfo import summary

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Append the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Now you can import your modules
from utils.imports import *
from models.StandAloneFFN import FFN
from models.StandAloneHyena import HyenaOperator
from data.datagenerator import DataGenerate
from data.datagenerator import task2
from Preprocessing.ISS import LowRankModel
from pytorch_lightning.callbacks import ModelCheckpoint

from functools import partial
from datetime import datetime


sweep_config = {
    'name' : 'sweep',

    'method': 'random',

    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},

    'parameters': {

        # 'order': {'values': [0, 2, 4, 8, 12, 16]},
        'order': {'values': [0]},
 
        'words': {'values': [2, 4, 6, 8, 12, 16, 20, 24]},
        
        'learning_rate': {'values': [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.0075, 0.01]},
       
        'epochs': {'values': [10, 20, 30, 40, 50, 100, 200]},

        # 'batch_size': {'values': [20, 40, 60, 80]}
        'batch_size': {'values': [1, 2, 4, 8, 16, 32, 64, 128]}
    }
}


# Define your model class
class ISSHClassifier(pl.LightningModule):
    def __init__(self, LowRank, Hyena, FFN, input, target, learning_rate, batch_size, overfit=False):
        super(ISSHClassifier, self).__init__()
        self.LowRank = LowRank
        self.Hyena = Hyena
        self.FFN = FFN
        self.x_data = input
        self.target = target
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.overfit = overfit

    def forward(self, x):
        y = self.LowRank(x)
        y = y.permute(0, 2, 1)
        if self.Hyena:
            y = self.Hyena(y)
        y = y.permute(0, 2, 1)
        z = y[:, :, -1]
        out = self.FFN(z)
        return out

    def prepare_data(self):
        total_dataset = TensorDataset(self.x_data, self.target)
        if self.overfit:
            self.train_dataset = total_dataset
            self.test_dataset = total_dataset
        else:
            train_size = int(0.8 * len(total_dataset))
            test_size = len(total_dataset) - train_size
            self.train_dataset, self.test_dataset = random_split(total_dataset, [train_size, test_size])

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        return train_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)

        _, predicted_labels = torch.max(output, dim=1)
        correct = (predicted_labels == target).sum().item()
        accuracy = correct / target.size(0)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)

        loss = F.cross_entropy(output, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)

        _, predicted_labels = torch.max(output, dim=1)
        correct = (predicted_labels == target).sum().item()
        accuracy = correct / target.size(0)
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)
        return {'test_accuracy': accuracy}

# Training function to use wandb sweep
def train(use_wandb=True, words=2, learning_rate=0.001, epochs=10, batch_size=20, overfit=False):
    if use_wandb:
        wandb.init()
        config = wandb.config
        order = config.order
        words = config.words
        learning_rate = config.learning_rate
        epochs = config.epochs
        batch_size = config.batch_size

    # Convert data into tensors
    x_datax = torch.tensor([[[*idata] for idata in zip(*data_point[:-1])] for data_point in data], dtype=torch.float32)
    x_data = x_datax.permute(0, 2, 1)
    labels = torch.tensor([data_point[-1] for data_point in data])

    # Assigning Values
    d_model = config.words

    # Initialize the model with sweep parameters
    LowRank = LowRankModel(words)

    if order == 0:
        Hyena = None
    else:
        Hyena = HyenaOperator(d_model=d_model, l_max=sequence_length, order=order, dropout=0.0, filter_dropout=0.0)
    ffn = FFN(words)
    model = ISSHClassifier(LowRank, Hyena, ffn, x_data, labels, learning_rate, batch_size, overfit=overfit)
    # wandb_logger.watch seems to have a bug; it does not always log the graph; so, we print the model here (-> Logs in wandb)
    print(model)

    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')

    if use_wandb:
        wandb_logger = WandbLogger(project='ISSHJumps', log_model="all")
        wandb_logger.watch(model, log="all", log_freq=1 if overfit else 10, log_graph=True)
    else:
        wandb_logger = None

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=epochs,
        logger=wandb_logger,
        log_every_n_steps=1 if overfit else 10,
        default_root_dir=current_dir,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model)
    trainer.test(model)

OVERFIT = False
OVERFIT = True

if OVERFIT:
    Total_batches = 10
    sweep_config['parameters']['batch_size']['values'] = [10]
else:
    Total_batches = 1000

sequence_length = 500
dim = 2
jumps = 3
data = task2(Total_batches, sequence_length, jumps)

def run_sweep():
    wandb.login() # Use wandb login procedure, instead of hardcoded API key.
    sweep_config['name'] +=  datetime.now().strftime('-%Y-%m-%d-%H:%M')
    if OVERFIT:
        sweep_config['name'] += '-overfit'
    sweep_id = wandb.sweep(sweep_config, project='ISSHJumps')
    wandb.agent(sweep_id, function=partial(train,overfit=OVERFIT), count=100)

if __name__ == "__main__":
    run_sweep()
    # train(use_wandb=False)