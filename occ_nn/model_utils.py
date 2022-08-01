import torch
from torch import nn
import torch.nn.functional as F

class F_model(nn.Module):
    def __init__(self, m, u, hidden):
        super().__init__()
        
        layers = []
        in_channel = m
        for i, out_channel in enumerate(hidden):
            layers.append(nn.Linear(in_channel, out_channel))
            layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.LeakyReLU(0.2) if i!=0 else nn.Tanh())
            in_channel = out_channel
        layers.append(nn.Linear(in_channel, u))

        self.block = nn.Sequential(*layers)
        self.m, self.u = m, u
    
    def forward(self, x):
        # [b,m+1,m]
        batch = x.shape[0]
        latent = self.block(x.view(-1,self.m)).view(batch, -1, self.u)
        # [b,m+1,u]
        norm_latent = F.normalize(latent,dim=1,p=2)
        return F.normalize(norm_latent,dim=2,p=2)

class G_model(nn.Module):
    def __init__(self, k, u, hidden):
        super().__init__()
        
        layers = []
        in_channel = k
        for i, out_channel in enumerate(hidden):
            layers.append(nn.Linear(in_channel, out_channel))
            if i>0:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.LeakyReLU(0.2)) # TODO change to sin/cos
            in_channel = out_channel
        layers.append(nn.Linear(in_channel, u))

        self.block = nn.Sequential(*layers)
        self.k, self.u = k, u
    
    def forward(self, x):
        # [b,m+1,k]
        batch = x.shape[0]
        latent = self.block(x.view(-1,self.k)).view(batch, -1, self.u)
        # [b,m+1,u]
        norm_latent = F.normalize(latent,dim=1,p=2)
        return F.normalize(norm_latent,dim=2,p=2)