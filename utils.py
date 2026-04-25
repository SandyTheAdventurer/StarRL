import torch
import torch.nn as nn
from pathlib import Path
from typing import Any
import numpy as np
from matplotlib import pyplot as plt
from torch.distributions import Normal, Categorical, Bernoulli

def evaluate_agent(metrics: dict) -> dict:
    time_min = max(metrics['game_time'] / 60, 1)
    
    eco = metrics['economic']
    eco_pts = (
        (min(eco['mineral_collection_efficiency'], 5) / 5 * 10) +
        (max(10 - (eco['idle_worker_time'] / time_min), 0) * 2) +
        (max(10 - (eco['idle_production_time'] / time_min), 0) * 2)
    )
    
    mil = metrics['military']
    mil_pts = (
        (min(mil['damage_ratio'], 2) / 2 * 20) +
        (min(mil['kill_value_ratio'], 1.5) / 1.5 * 20)
    )
    
    res = metrics['resources']
    prod = metrics['production']
    macro_pts = (
        (res['resource_spending_rate'] * 15) +
        (min(prod['net_value_retained'] / 5000, 1) * 15)
    )

    prod_pts = (
        (min(prod['total_value_created'] / 5000, 1) * 15) +
        (min(prod['net_value_retained'] / 5000, 1) * 15) +
        (min(prod['value_lost_structures'] / 1000, 1) * 10) +
        (20 if prod['total_structure_value'] > 0 else 0)
    )

    scores = {
        "economic_score": round(eco_pts, 2),
        "military_score": round(mil_pts, 2),
        "macro_score": round(macro_pts, 2),
        "production_score": round(prod_pts, 2)
    }
    return scores

def update_elo(player_elo, opponent_elo, actual_score, elo_k = 32):
    expected = 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))
    return player_elo + elo_k * (actual_score - expected)

def plot_radar_chart(scores: dict, output_path: str = "performance_radar.png"):
    categories = ['Economic', 'Military', 'Macro', 'Production']
    values = [
        scores['economic_score'], 
        scores['military_score'], 
        scores['macro_score'],
        scores.get('production_score', 0) 
    ]
    
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], categories)
    
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)
    
    ax.set_ylim(0, 100)
    
    plt.title("Agent Performance Profile", size=15, y=1.1)
    plt.savefig(output_path)
    plt.close()

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