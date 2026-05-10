import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import physics_tools as pt
from materials import Silicon, q







# Create the Neural Network

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 1)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        return self.output(x)


# generate data

def generate_training_data(filename="silicon_fermi_dataset.csv", grid_size=100):
    if os.path.exists(filename):
        print(f"[*] Dataset '{filename}' already exists. Skipping generation.")
        return filename