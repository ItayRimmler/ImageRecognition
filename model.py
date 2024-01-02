import torch as t
from torch import nn
import numpy as np

class Model(nn.Module):

    def __init__(self, inputsSize, outputSize, hiddenSize=640):
        super().__init__()
        self.lay1 = nn.Linear(in_features=inputsSize, out_features=round(hiddenSize))
        self.relu = nn.ReLU()
        self.lay2 = nn.Linear(in_features=round(hiddenSize), out_features=outputSize)

    def forward(self, inputs):
        processing = self.lay1(inputs)
        processing = self.relu(processing)
        outputs = self.lay2(processing)
        return outputs
