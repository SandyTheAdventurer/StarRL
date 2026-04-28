import torch
import torch.nn as nn
from pathlib import Path
from typing import Any
import numpy as np
from matplotlib import pyplot as plt
from torch.distributions import Normal, Categorical, Bernoulli

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _saturating_score(value: float, scale: float) -> float:
    if value <= 0:
        return 0.0
    return value / (value + scale)


def evaluate_agent(metrics: dict) -> dict:
    time_min = max(metrics.get("game_time", 0.0) / 60.0, 1.0)

    eco = metrics["economic"]
    mil = metrics["military"]
    res = metrics["resources"]
    prod = metrics["production"]

    mineral_eff = _clamp(eco.get("mineral_collection_efficiency", 0.0), 0.0, 5.0) / 5.0
    idle_worker_rate = _clamp(eco.get("idle_worker_time", 0.0) / time_min, 0.0, 60.0)
    idle_prod_rate = _clamp(eco.get("idle_production_time", 0.0) / time_min, 0.0, 60.0)

    eco_pts = (
        mineral_eff * 40.0
        + (1.0 - idle_worker_rate / 60.0) * 30.0
        + (1.0 - idle_prod_rate / 60.0) * 30.0
    )

    damage_ratio = _clamp(mil.get("damage_ratio", 0.0), 0.0, 3.0) / 3.0
    kill_value_ratio = _clamp(mil.get("kill_value_ratio", 0.0), 0.0, 2.0) / 2.0
    damage_per_min = (mil.get("total_damage_dealt", 0.0) / time_min)
    damage_rate_score = _saturating_score(damage_per_min, 500.0)

    mil_pts = (
        damage_ratio * 40.0
        + kill_value_ratio * 40.0
        + damage_rate_score * 20.0
    )

    spending_rate = _clamp(res.get("resource_spending_rate", 0.0), 0.0, 1.2) / 1.2
    retained_score = _saturating_score(prod.get("net_value_retained", 0.0), 4000.0)
    macro_pts = spending_rate * 60.0 + retained_score * 40.0

    created_score = _saturating_score(prod.get("total_value_created", 0.0), 5000.0)
    structure_score = _saturating_score(prod.get("total_structure_value", 0.0), 1500.0)
    loss_penalty = _clamp(prod.get("value_lost_structures", 0.0) / 2000.0, 0.0, 1.0)
    prod_pts = created_score * 45.0 + structure_score * 35.0 + (1.0 - loss_penalty) * 20.0

    scores = {
        "economic_score": round(_clamp(eco_pts, 0.0, 100.0), 2),
        "military_score": round(_clamp(mil_pts, 0.0, 100.0), 2),
        "macro_score": round(_clamp(macro_pts, 0.0, 100.0), 2),
        "production_score": round(_clamp(prod_pts, 0.0, 100.0), 2),
    }
    return scores

def update_elo(player_elo, opponent_elo, actual_score, elo_k = 32):
    expected = 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))
    return player_elo + elo_k * (actual_score - expected)

def plot_radar_chart(scores: dict, output_path: str = "performance_radar.png"):
    categories = ["Economic", "Military", "Macro", "Production"]
    values = [
        scores.get("economic_score", 0.0),
        scores.get("military_score", 0.0),
        scores.get("macro_score", 0.0),
        scores.get("production_score", 0.0),
    ]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#666666", size=9)
    ax.grid(color="#dddddd", linewidth=0.8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    ax.plot(angles, values, linewidth=2, linestyle="solid", color="#1f77b4")
    ax.fill(angles, values, color="#1f77b4", alpha=0.15)

    plt.title("Agent Performance Profile", size=14, y=1.08)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
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