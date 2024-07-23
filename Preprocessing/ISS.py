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
from Preprocessing.LowRank import ISS
from data.datagenerator import SimpleDataGenerate

class FFN(pl.LightningModule):
    def __init__(self, input_dim, input, target):
        super(FFN, self).__init__()
        self.x_input = input
        self.target = target
        self.fc1 = nn.Linear(input_dim, 5)
        self.fc2 = nn.Linear(5, 3)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, 32)
        # self.fc5 = nn.Linear(32, 3)

    def forward(self, x):
        # y = model(x)
        # x = rearrange(y, 'b l d -> b (l d)')
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc4(x))
        x = self.fc2(x)
        return x

    def prepare_data(self):
        total_dataset = TensorDataset(self.x_input, self.target)
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
        Array_accuracy.append(predicted_labels)
        Actual_labels.append(y)

        # Log accuracy
        self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)

        return {'test_accuracy': accuracy}

Total_batches = 1000
sequence_length = 500
dim = 2
data = DataGenerate(Total_batches, sequence_length, dim)
# data = SimpleDataGenerate(Total_batches, sequence_length)

x_datax = torch.tensor([[[*idata] for idata in zip(*data_point[:-1])] for data_point in data])
x_data = x_datax.permute(0, 2, 1)
x_data = torch.tensor(x_data, dtype=torch.float32)
labels = torch.tensor([data_point[-1] for data_point in data])
words = 2

bs, b, c = x_data.size()
x = torch.zeros([bs, words])

for i in range(words):
    x[:, i] = ISS(x_data, i)[-1][-1][:,-1:].view(-1)

Array_accuracy = []
Actual_labels = []

# model = LowRankModel(words)
input_dim = words
# print(x)
# print(labels)
FFNmodel = FFN(input_dim, x, labels)

checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=11,
    log_every_n_steps=10,
    default_root_dir = current_dir
)

trainer.fit(FFNmodel)
trainer.test(FFNmodel)



