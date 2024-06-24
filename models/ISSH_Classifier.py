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
# from Preprocessing.ISS import LowRankModel
from models.LowRank import LowRankModel
from pytorch_lightning.callbacks import ModelCheckpoint

Total_batches = 1000
sequence_length = 500
dim = 2
data = DataGenerate(Total_batches, sequence_length, dim)
device = torch.device("cuda")

USE_WANDB = True
if USE_WANDB:
    wandb.login(key = '7e169996e30d15f253a5f1d92ef090f2f3b948c4')

# ### FunctionClassifier
# class ISSHClassifier(pl.LightningModule):
#     def __init__(
#             self,
#             LowRank,
#             HyenaOperator,
#             FFN,
#             input,
#             target
#         ):
#         super(ISSHClassifier, self).__init__()
#         self.Hyena = HyenaOperator
#         self.FFN = FFN
#         self.x_data = input
#         self.target = target

#     def forward(self,x):
#         y = LowRank(x)
#         # p = y.permute(0, 2, 1)
#         # q = self.Hyena(p)
#         z = y[:,:,-1]
#         out = self.FFN(z)
#         return out

#     def prepare_data(self):
#         total_dataset = TensorDataset(self.x_data, self.target)
#         train_size = int(0.8 * len(total_dataset))
#         test_size = len(total_dataset) - train_size
#         self.train_dataset, self.test_dataset = random_split(total_dataset, [train_size, test_size])

#     def train_dataloader(self):
#         train_loader = DataLoader(self.train_dataset, batch_size=40, shuffle=True, num_workers=23)
#         return train_loader

#     def test_dataloader(self):
#         test_loader = DataLoader(self.test_dataset, batch_size=40, shuffle=False, num_workers=23)
#         return test_loader

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
#         # scheduler = StepLR(optimizer, step_size=1) 
#         # "lr_scheduler": scheduler
#         return {"optimizer": optimizer}

#     def training_step(self, batch, batch_idx):
#         data, target = batch
#         output = self.forward(data)

#         # Compute training accuracy
#         _, predicted_labels = torch.max(output, dim=1)
#         correct = (predicted_labels == target).sum().item()
#         accuracy = correct / target.size(0)

#         # Log training accuracy
#         self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger = True)

#         loss = F.cross_entropy(output, target)
#         self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         # return loss
#         return {'loss': loss}

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)

#         # Compute predicted labels
#         _, predicted_labels = torch.max(y_hat, dim=1)

#         # Compute accuracy
#         correct = (predicted_labels == y).sum().item()
#         total = y.size(0)
#         accuracy = correct / total
#         Array_accuracy.append(predicted_labels)
#         Actual_labels.append(y)

#         # Log accuracy
#         self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)

#         return {'test_accuracy': accuracy}
    
# # Convert data into tensors
# x_datax = torch.tensor([[[*idata] for idata in zip(*data_point[:-1])] for data_point in data])
# x_data = x_datax.permute(0, 2, 1)
# x_data = torch.tensor(x_data, dtype=torch.float32)
# labels = torch.tensor([data_point[-1] for data_point in data])

# Array_accuracy = []
# Actual_labels = []

# # Assigning Values
# d_model = dim
# words = 4
# # batch_size = 1000
# # train_ratio = 0.8

# # Initialize HyenaOperator model
# LowRank = LowRankModel(words)
# ffn = FFN(words)
# Hyena = HyenaOperator(d_model=d_model, l_max=sequence_length, order=8, dropout=0.0, filter_dropout=0.0)
# Model = ISSHClassifier(LowRank, Hyena, ffn, x_data, labels)

# checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')

# # initiate wandb logger
# if USE_WANDB:
#     wandb_logger = WandbLogger(project = 'ISS',log_model="all")
# else:
#     wandb_logger = None

# # Initialize Trainer and start training
# trainer = pl.Trainer(
#     accelerator="gpu",
#     max_epochs=11,
#     logger = wandb_logger,
#     log_every_n_steps=10,
#     default_root_dir = current_dir,
#     callbacks=[checkpoint_callback]
# )

# # print('a')
# # summary(Model, input_size = x_data.size())
# # print('e')

# if USE_WANDB:
#     wandb_logger.watch(Model, log="gradients", log_freq=50, log_graph=True)

# trainer.fit(Model)
# trainer.test(Model)

# if USE_WANDB:
#     wandb.finish()




sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'words': {'values': [2, 4, 6, 8, 10, 12, 14, 16]},
        'learning_rate': {'values': [0.0001, 0.0005, 0.001, 0.002, 0.005]},
        'epochs': {'values': [10, 20]},
        'batch_size': {'values': [20, 40, 60, 80]}
    }
}

sweep_id = wandb.sweep(sweep_config, project='ISS')

# Define your model class
class ISSHClassifier(pl.LightningModule):
    def __init__(self, LowRank, HyenaOperator, FFN, input, target, learning_rate, batch_size):
        super(ISSHClassifier, self).__init__()
        self.LowRank = LowRank
        self.Hyena = HyenaOperator
        self.FFN = FFN
        self.x_data = input
        self.target = target
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def forward(self, x):
        print(x.device)
        y = self.LowRank(x)
        z = y[:, :, -1]
        out = self.FFN(z)
        return out

    def prepare_data(self):
        total_dataset = TensorDataset(self.x_data, self.target)
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
def train():
    wandb.init()
    config = wandb.config

    # Convert data into tensors
    x_datax = torch.tensor([[[*idata] for idata in zip(*data_point[:-1])] for data_point in data], dtype=torch.float32)
    x_data = x_datax.permute(0, 2, 1)
    labels = torch.tensor([data_point[-1] for data_point in data])

    # Assigning Values
    d_model = dim

    # Initialize the model with sweep parameters
    LowRank = LowRankModel(config.words)
    Hyena = HyenaOperator(d_model=d_model, l_max=sequence_length, order=8, dropout=0.0, filter_dropout=0.0)
    ffn = FFN(config.words)
    model = ISSHClassifier(LowRank, Hyena, ffn, x_data, labels, config.learning_rate, config.batch_size)

    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')

    wandb_logger = WandbLogger(project='ISS', log_model="all")

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=config.epochs,
        logger=wandb_logger,
        log_every_n_steps=10,
        default_root_dir=current_dir,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model)
    trainer.test(model)

if __name__ == "__main__":
    wandb.agent(sweep_id, function=train, count=10)