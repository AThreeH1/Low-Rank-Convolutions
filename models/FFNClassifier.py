import sys
sys.path.append('c:\\Study\\DS\\Classifier')

from utils.imports import *
from models.StandAloneFFN import FFN
from data.datagenerator import data

USE_WANDB = False
if USE_WANDB:
    wandb.login(key = ['7e169996e30d15f253a5f1d92ef090f2f3b948c4'])

class FFNClassifier(pl.LightningModule):
    def __init__(
            self,
            ffn
    ):
        super(FFNClassifier, self).__init__()
        self.ffn = ffn

    def forward(self,x):
        y = rearrange(x, 'b l d -> b (l d)')
        out = self.ffn(y)
        return out

    def prepare_data(self):
        total_dataset = TensorDataset(x_data, labels)
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
        # _, yn = torch.max(y, dim=1)

        # Compute accuracy
        correct = (predicted_labels == y).sum().item()
        total = y.size(0)
        accuracy = correct / total
        Array_accuracy.append(predicted_labels)
        Actual_labels.append(y)

        # Log accuracy
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)

        return {'test_accuracy': accuracy}



# device = torch.device("cuda")

# Convert data into tensors
x_datax = torch.tensor([[[a, b] for a, b in zip(idata[0], idata[1])] for idata in data])
x_data = x_datax.permute(0, 2, 1)
x_data = torch.tensor(x_data, dtype=torch.float32)
labels = torch.tensor([label for _, _, label in data])

Array_accuracy = []
Actual_labels = []

# Initialize HyenaOperator model
ffn = FFN()
Model = FFNClassifier(ffn)

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
    logger= wandb_logger,
    log_every_n_steps=1
)

if USE_WANDB:
    wandb_logger.watch(Model, log="gradients", log_freq=50, log_graph=True)

trainer.fit(Model)
trainer.test()

if USE_WANDB:
    wandb.finish()