import torch
import torch.nn as nn
import math
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
    avg = metrics.get("averages", {})

    avg_income = _clamp(avg.get("income_rate", 0.0), 0.0, 1.0)
    avg_workers = _clamp(avg.get("workers", 0.0), 0.0, 60.0) / 60.0
    avg_supply_used = float(avg.get("supply_used", 0.0))
    avg_supply_cap = max(float(avg.get("supply_cap", 0.0)), 1.0)
    avg_supply_util = _clamp(avg_supply_used / avg_supply_cap, 0.0, 1.0)
    avg_tech = _clamp(avg.get("tech_level", 0.0), 0.0, 2.0) / 2.0
    avg_army_supply = max(avg_supply_used - float(avg.get("workers", 0.0)), 0.0)
    avg_army_score = _clamp(avg_army_supply / 100.0, 0.0, 1.0)
    avg_structures = _clamp(avg.get("structures", 0.0), 0.0, 20.0) / 20.0

    mineral_eff = _clamp(eco.get("mineral_collection_efficiency", 0.0), 0.0, 5.0) / 5.0
    idle_worker_rate = _clamp(eco.get("idle_worker_time", 0.0) / time_min, 0.0, 60.0)
    idle_prod_rate = _clamp(eco.get("idle_production_time", 0.0) / time_min, 0.0, 60.0)

    eco_pts = (
        mineral_eff * 40.0
        + (1.0 - idle_worker_rate / 60.0) * 30.0
        + (1.0 - idle_prod_rate / 60.0) * 30.0
        + avg_income * 10.0
        + avg_workers * 5.0
    )

    damage_ratio = _clamp(mil.get("damage_ratio", 0.0), 0.0, 3.0) / 3.0
    kill_value_ratio = _clamp(mil.get("kill_value_ratio", 0.0), 0.0, 2.0) / 2.0
    damage_per_min = (mil.get("total_damage_dealt", 0.0) / time_min)
    damage_rate_score = _saturating_score(damage_per_min, 500.0)

    mil_pts = (
        damage_ratio * 40.0
        + kill_value_ratio * 40.0
        + damage_rate_score * 20.0
        + avg_army_score * 10.0
    )

    spending_rate = _clamp(res.get("resource_spending_rate", 0.0), 0.0, 1.2) / 1.2
    retained_score = _saturating_score(prod.get("net_value_retained", 0.0), 4000.0)
    macro_pts = (
        spending_rate * 60.0
        + retained_score * 40.0
        + avg_supply_util * 5.0
        + avg_tech * 10.0
    )

    created_score = _saturating_score(prod.get("total_value_created", 0.0), 5000.0)
    structure_score = _saturating_score(prod.get("total_structure_value", 0.0), 1500.0)
    loss_penalty = _clamp(prod.get("value_lost_structures", 0.0) / 2000.0, 0.0, 1.0)
    prod_pts = (
        created_score * 45.0
        + structure_score * 35.0
        + (1.0 - loss_penalty) * 20.0
        + avg_structures * 5.0
    )

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
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layers,
        act=nn.ReLU,
        layer_norm: bool = False,
        init_orthogonal: bool = False,
        output_gain: float = 1.0,
        hidden_gain: float | None = None,
    ):
        super(MLP, self).__init__()
        assert n_layers >= 2, "MLP must have at least two layers"
        if hidden_gain is None:
            hidden_gain = math.sqrt(2.0)

        layers = []
        in_dim = input_dim
        for layer_idx in range(n_layers):
            out_dim = output_dim if layer_idx == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx != n_layers - 1:
                if layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(act())
            in_dim = out_dim

        self.model = nn.Sequential(*layers)

        if init_orthogonal:
            linears = [m for m in self.model if isinstance(m, nn.Linear)]
            last_idx = len(linears) - 1
            for idx, layer in enumerate(linears):
                gain = output_gain if idx == last_idx else hidden_gain
                nn.init.orthogonal_(layer.weight, gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.model(x)

class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4):
        self.mean = torch.zeros((), dtype=torch.float64)
        self.var = torch.ones((), dtype=torch.float64)
        self.count = float(epsilon)

    def update(self, x: torch.Tensor):
        if x.numel() == 0:
            return
        x = x.detach().to(dtype=torch.float64, device="cpu")
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = x.numel()
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total_count
        new_var = m2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = float(total_count)

    def normalize(self, x: torch.Tensor, eps: float = 1e-8):
        mean = self.mean.to(dtype=x.dtype, device=x.device)
        var = self.var.to(dtype=x.dtype, device=x.device)
        return (x - mean) / torch.sqrt(var + eps)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
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
        self.act = nn.SiLU()
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
        log_probs = dist.log_prob(sample)
        return sample, log_probs

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

class EntityEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, n_heads=4, n_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

    def forward(self, x, mask):

        x = self.input_proj(x)

        key_padding_mask = (mask == 0)

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        mask = mask.unsqueeze(-1)
        x = x * mask

        pooled = x.sum(dim=1) / (mask.sum(dim=1) + 1e-6)

        return pooled
    
class RolloutBuffer:
    def __init__(self, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        self.images = []
        self.resources = []
        self.entities = []
        self.masks = []
        self.action_masks = []

        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

        self.h_states = []
        self.c_states = []

    def add(self, obs, action, log_prob, reward, done, value, lstm_state, action_mask=None):
        img, res, ent, mask = obs
        self.images.append(torch.as_tensor(img, dtype=torch.float16, device=self.device))
        self.resources.append(torch.as_tensor(res, dtype=torch.float16, device=self.device))
        self.entities.append(torch.as_tensor(ent, dtype=torch.float16, device=self.device))
        self.masks.append(torch.as_tensor(mask, dtype=torch.float16, device=self.device))
        if action_mask is not None:
            self.action_masks.append(torch.as_tensor(action_mask, dtype=torch.bool, device=self.device))

        self.actions.append(torch.as_tensor(action, dtype=torch.long, device=self.device))
        self.log_probs.append(log_prob.detach().to(self.device))
        self.rewards.append(torch.as_tensor(reward, dtype=torch.float32, device=self.device))
        self.dones.append(torch.as_tensor(done, dtype=torch.float32, device=self.device))
        self.values.append(value.detach().to(self.device))

        h, c = lstm_state
        self.h_states.append(h.detach().to(self.device).squeeze(0))
        self.c_states.append(c.detach().to(self.device).squeeze(0))

class CrossAttentionFusion(nn.Module):
    def __init__(self, image_dim, resource_dim, entity_dim, fusion_dim, n_heads):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.resource_proj = nn.Linear(resource_dim, fusion_dim)
        self.entity_proj = nn.Linear(entity_dim, fusion_dim)
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.attn = nn.MultiheadAttention(
            fusion_dim,
            n_heads,
            batch_first=True,
        )
        nn.init.normal_(self.fusion_token, mean=0.0, std=0.02)

    def forward(self, image_features, resource_features, entity_features):
        tokens = torch.stack(
            [
                self.image_proj(image_features),
                self.resource_proj(resource_features),
                self.entity_proj(entity_features),
            ],
            dim=1,
        )
        query = self.fusion_token.expand(tokens.size(0), -1, -1)
        attn_out, _ = self.attn(query, tokens, tokens, need_weights=False)
        return (query + attn_out).squeeze(1)