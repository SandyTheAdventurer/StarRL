import torch
import torch.nn as nn
from pathlib import Path
from typing import Any
from torch.distributions import Normal, Categorical, Bernoulli

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, act=nn.SiLU, optimizer = "Adam", lr=0.001):
        super(MLP, self).__init__()
        assert n_layers >= 2, "MLP must have at least two layers"
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_layers - 2):
            layers.append(act(inplace=True))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(act(inplace=True))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
        match optimizer:
            case "Adam":
                self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            case "SGD":
                self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
            case _:
                raise ValueError(f"Unsupported optimizer: {optimizer}")

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.act(out)
        return out
    
class ImageFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImageFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.resblock1 = ResBlock(out_channels, out_channels)
        self.resblock2 = ResBlock(out_channels, out_channels)
        self.resblock3 = ResBlock(out_channels, out_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.pool(out)
        return out.flatten(1)

class ContinuousHead(nn.Module):
    def __init__(self, input_dim):
        super(ContinuousHead, self).__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        mean, log_std = self.linear(x).chunk(2, dim=-1)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        sample = dist.sample()
        logit = dist.log_prob(sample)
        return sample, logit
    
class DiscreteHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DiscreteHead, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        logits = self.linear(x)
        dist = Categorical(logits=logits)
        sample = dist.sample()
        logit = dist.log_prob(sample)
        return sample, logit

class MultiBinaryHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        logits = self.linear(x)
        dist = Bernoulli(logits=logits)
        sample = dist.sample()
        log_prob = dist.log_prob(sample)

        return sample, log_prob