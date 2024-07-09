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
from models.PathDev import PathDevelopmentNetwork
from data.datagenerator import DataGenerate
from data.datagenerator import SimpleDataGenerate
from pytorch_lightning.callbacks import ModelCheckpoint

Total_batches = 1000
sequence_length = 500
dim = 2
data = DataGenerate(Total_batches, sequence_length, dim)
device = torch.device("cuda")

USE_WANDB = True
if USE_WANDB:
    wandb.login(key = '7e169996e30d15f253a5f1d92ef090f2f3b948c4')

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': {
        
        'learning_rate': {'values': [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.0075, 0.01]},
        
        'epochs': {'values': [10, 20]},

        'batch_size': {'values': [20, 40, 60, 80]}
        
    }
}

sweep_id = wandb.sweep(sweep_config, project='PathDev')

### FunctionClassifier
class PathClassifier(pl.LightningModule):
    def __init__(
            self,
            PathD,
            FFN,
            input,
            target,
            learning_rate,
            batch_size
        ):
        super(PathClassifier, self).__init__()
        self.FFN = FFN
        self.PathD = PathD
        self.x_data = input
        self.target = target
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def forward(self,x):
        y = self.PathD(x)
        # p = y.permute(0, 2, 1)
        # q = self.Hyena(p)
        a, b, _, _ = y.shape
        k = y.view(a, b, 9)
        z = k[:,-1,:]
        z1 = z.view(a, 9)
        zf = z1.float()
        out = self.FFN(zf)
        return out

    def prepare_data(self):
        total_dataset = TensorDataset(self.x_data, self.target)
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
        optimizer = torch.optim.Adam(self.parameters(), lr= self.learning_rate)
        # scheduler = StepLR(optimizer, step_size=1) 
        # "lr_scheduler": scheduler
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)

        # Compute training accuracy
        _, predicted_labels = torch.max(output, dim=1)
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
    
def train():
    wandb.init()
    config = wandb.config

    # Convert data into tensors
    x_datax = torch.tensor([[[*idata] for idata in zip(*data_point[:-1])] for data_point in data])
    x_data = torch.tensor(x_datax, dtype=torch.float32)
    labels = torch.tensor([data_point[-1] for data_point in data])

    Array_accuracy = []
    Actual_labels = []

    # Assigning Values
    d_model = dim
    d = 3

    # Initialize HyenaOperator model
    PathD = PathDevelopmentNetwork(d)
    ffn = FFN(d**2)
    Model = PathClassifier(PathD, ffn, x_data, labels, config.learning_rate, config.batch_size)

    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')

    # initiate wandb logger
    if USE_WANDB:
        wandb_logger = WandbLogger(project = 'PathDev',log_model="all")
    else:
        wandb_logger = None

    # Initialize Trainer and start training
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=config.epochs,
        logger = wandb_logger,
        log_every_n_steps=10,
        default_root_dir = current_dir,
        callbacks=[checkpoint_callback]
    )

    # if USE_WANDB:
    #     wandb_logger.watch(Model, log="gradients", log_freq=50, log_graph=True)

    trainer.fit(Model)
    trainer.test(Model)

    # if USE_WANDB:
    #     wandb.finish()

if __name__ == "__main__":
    wandb.agent(sweep_id, function=train, count=10)
        