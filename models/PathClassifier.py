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
from models.LowRank import ISS
from data.datagenerator import SimpleDataGenerate

Total_batches = 1000
sequence_length = 500
dim = 2
data = DataGenerate(Total_batches, sequence_length, dim)
device = torch.device("cuda")

USE_WANDB = True
if USE_WANDB:
    wandb.login(key = '7e169996e30d15f253a5f1d92ef090f2f3b948c4')

### FunctionClassifier
class PathClassifier(pl.LightningModule):
    def __init__(
            self,
            LowRank,
            HyenaOperator,
            FFN,
            input,
            target
        ):
        super(PathClassifier, self).__init__()
        self.Hyena = HyenaOperator
        self.FFN = FFN
        self.x_data = input
        self.target = target

    def forward(self,x):
        y = LowRank(x)
        # p = y.permute(0, 2, 1)
        # q = self.Hyena(p)
        z = y[:,:,-1]
        out = self.FFN(z)
        return out

    def prepare_data(self):
        total_dataset = TensorDataset(self.x_data, self.target)
        train_size = int(0.8 * len(total_dataset))
        test_size = len(total_dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(total_dataset, [train_size, test_size])

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=40, shuffle=True, num_workers=23)
        return train_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=40, shuffle=False, num_workers=23)
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
x_datax = torch.tensor([[[*idata] for idata in zip(*data_point[:-1])] for data_point in data])
x_data = x_datax.permute(0, 2, 1)
x_data = torch.tensor(x_data, dtype=torch.float32)
labels = torch.tensor([data_point[-1] for data_point in data])

Array_accuracy = []
Actual_labels = []

# Assigning Values
d_model = dim
words = 4
# batch_size = 1000
# train_ratio = 0.8

# Initialize HyenaOperator model
LowRank = LowRankModel(words)
ffn = FFN(words)
Hyena = HyenaOperator(d_model=d_model, l_max=sequence_length, order=8, dropout=0.0, filter_dropout=0.0)
Model = ISSHClassifier(LowRank, Hyena, ffn, x_data, labels)

checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')

# initiate wandb logger
if USE_WANDB:
    wandb_logger = WandbLogger(project = 'ISS',log_model="all")
else:
    wandb_logger = None

# Initialize Trainer and start training
trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=11,
    logger = wandb_logger,
    log_every_n_steps=10,
    default_root_dir = current_dir,
    callbacks=[checkpoint_callback]
)

# print('a')
# summary(Model, input_size = x_data.size())
# print('e')

if USE_WANDB:
    wandb_logger.watch(Model, log="gradients", log_freq=50, log_graph=True)

trainer.fit(Model)
trainer.test(Model)

if USE_WANDB:
    wandb.finish()


model = PathDevelopmentNetwork(3)
tensor = torch.ones(1, 5, 9)
h = torch.ones(1,5)

reshaped_tensor = tensor.view(1, 5, 9)

#Split tensor along the third dimension into 9 tensors
sliced_tensors = torch.split(reshaped_tensor, 1, dim=2)

# Reshape each sliced tensor to (1, 5, 1)
final_tensors = [t.view(1, 5, 1) for t in sliced_tensors]

output = []

for i in range(len(final_tensors)):
    in_fft = torch.fft.fft(final_tensors[i].squeeze(), dim = -1)
    h_fft = torch.fft.fft(h, dim = -1)
    Mul = in_fft * h_fft
    out_fft = torch.fft.ifft(Mul, dim = -1)
    output.append(out_fft)

print(output)