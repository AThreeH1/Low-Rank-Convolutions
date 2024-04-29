from utils.imports import *

class FFN(nn.Module):
    def __init__(self, layer_sizes):
        super(FFN, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers[:-1])  # Exclude the last activation

    def forward(self, x):
        return self.model(x)