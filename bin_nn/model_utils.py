import torch
from torch import nn
import torch.nn.functional as F

class Binary_model(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        
        layers = []
        in_channel = d
        for i, out_channel in enumerate(hidden):
            layers.append(nn.Linear(in_channel, out_channel))
            layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.LeakyReLU(0.2))
            in_channel = out_channel
        layers.append(nn.Linear(in_channel, 1))

        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        # [b, d]
        return self.block(x).squeeze(1)
