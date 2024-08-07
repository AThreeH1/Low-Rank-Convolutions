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
from data.datagenerator import DataGenerate, task2
from Preprocessing.ISS import ISS
from data.datagenerator import SimpleDataGenerate
from pytorch_lightning.callbacks import ModelCheckpoint

from functools import partial
from datetime import datetime

sweep_config = {
    'name' : 'sweep',

    'method': 'random',

    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},

    'parameters': {
 
        'words': {'values': [2, 4, 8, 16, 20, 24]},
        
        'learning_rate': {'values': [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.0075, 0.01]},
       
        'epochs': {'values': [10, 20, 40, 100, 200]}, 

        'batch_size': {'values': [16, 32, 64, 128]}, 

    }
}


class FFN(pl.LightningModule):
    def __init__(self, input_dim, input, target, learning_rate, batch_size):
        super(FFN, self).__init__()
        self.x_input = input
        self.target = target
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.fc1 = nn.Linear(input_dim, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def prepare_data(self):
        total_dataset = TensorDataset(self.x_input, self.target)
        train_size = int(0.8 * len(total_dataset))
        test_size = len(total_dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(total_dataset, [train_size, test_size])

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=23)
        return train_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=23)
        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)

        # Compute training accuracy
        _, predicted_labels = torch.max(output, dim=1)
        print(predicted_labels.size())
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

        # Compute accuracy
        correct = (predicted_labels == y).sum().item()
        total = y.size(0)
        accuracy = correct / total

        # Log accuracy
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)

        return {'test_accuracy': accuracy}


# Training function to use wandb sweep
def train(use_wandb=True, words=2, learning_rate=0.001, epochs=10, batch_size=20):
    if use_wandb:
        wandb.init()
        config = wandb.config
        words = config.words
        learning_rate = config.learning_rate
        epochs = config.epochs
        batch_size = config.batch_size

    # Convert data into tensors
    x_datax = torch.tensor([[[*idata] for idata in zip(*data_point[:-1])] for data_point in data], dtype=torch.float32)
    x_data = x_datax.permute(0, 2, 1)
    labels = torch.tensor([data_point[-1] for data_point in data])

    bs, b, c = x_data.size()
    x = torch.zeros([bs, words])

    for i in range(words):
        x[:, i] = ISS(x_data, i)[-1][-1][:,-1:].view(-1)

    model = FFN(words, x, labels, learning_rate, batch_size)
    print(model)

    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')

    if use_wandb:
        wandb_logger = WandbLogger(project='ISSJumps', log_model="all")
        wandb_logger.watch(model, log="all", log_freq=10, log_graph=True)
    else:
        wandb_logger = None

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=epochs,
        logger=wandb_logger,
        default_root_dir=current_dir,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model)
    trainer.test(model)



Total_batches = 1000
sequence_length = 500
jumps = 1
data = task2(Total_batches, sequence_length, jumps)

def run_sweep():
    wandb.login() # Use wandb login procedure, instead of hardcoded API key.
    sweep_config['name'] +=  'J' + str(jumps)
    sweep_config['name'] +=  datetime.now().strftime('-%Y-%m-%d-%H:%M')
    sweep_id = wandb.sweep(sweep_config, project='ISSJumps')
    wandb.agent(sweep_id, function=partial(train), count=100)

if __name__ == "__main__":
    run_sweep()

