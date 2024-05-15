import os
import sys

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

Total_batches = 1000
sequence_length = 500
dim = 4
data = DataGenerate(Total_batches, sequence_length, dim)

USE_WANDB = False
if USE_WANDB:
    wandb.login(key = ['7e169996e30d15f253a5f1d92ef090f2f3b948c4'])

### FunctionClassifier

class HyenaClassifier(pl.LightningModule):
    def __init__(
            self,
            HyenaOperator,
            FFN,
            input,
            target
        ):
        super(HyenaClassifier, self).__init__()
        self.Hyena = HyenaOperator
        self.FFN = FFN
        self.x_data = input
        self.target = target

    def forward(self,x):
        y = self.Hyena(x)
        y = x
        z = rearrange(y, 'b l d -> b (l d)')
        out = self.FFN(z)
        return out

    def prepare_data(self):
        total_dataset = TensorDataset(self.x_data, self.target)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
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
        Array_accuracy.append(predicted_labels)
        Actual_labels.append(y)

        # Log accuracy
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)

        return {'test_accuracy': accuracy}
    

# Convert data into tensors
x_datax = torch.tensor([[[a, b, c, d] for a, b, c, d in zip(idata[0], idata[1], idata[2], idata[3])] for idata in data])
x_data = x_datax.permute(0, 1, 2)
x_data = torch.tensor(x_data, dtype=torch.float32)
labels = torch.tensor([label for _, _, _, _, label in data])

Array_accuracy = []
Actual_labels = []

# Assigning Values
d_model = 4
batch_size = 1000
train_ratio = 0.8

# Initialize HyenaOperator model
ffn = FFN(sequence_length, dim)
Hyena = HyenaOperator(d_model=d_model, l_max=sequence_length, order=2, dropout=0.0, filter_dropout=0.0)
Model = HyenaClassifier(Hyena, ffn, x_data, labels)

checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')

# initiate wandb logger
if USE_WANDB:
    wandb_logger = WandbLogger(project = 'lightning_model',log_model="all")
else:
    wandb_logger = None

# Initialize Trainer and start training
trainer = pl.Trainer(
    accelerator="cpu",
    max_epochs=11,
    logger = wandb_logger,
    log_every_n_steps=1,
)

if USE_WANDB:
    wandb_logger.watch(Model, log="gradients", log_freq=50, log_graph=True)

trainer.fit(Model)
trainer.test()

if USE_WANDB:
    wandb.finish()