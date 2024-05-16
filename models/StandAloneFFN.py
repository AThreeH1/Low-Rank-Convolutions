import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Append the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utils.imports import *

class FFN(nn.Module):
    def __init__(self, sequence_length, dim):
        super(FFN, self).__init__()
        self.sequence_length = sequence_length
        self.dim = dim
        self.fc1 = torch.nn.Linear(dim, 3)
        # self.fc2 = torch.nn.Linear(256, 128)
        # self.fc3 = torch.nn.Linear(128, 64)
        # self.fc4 = torch.nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        # x = torch.softmax(self.fc4(x), dim=1)
        return x

