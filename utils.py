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
    
class Observer():
    def __init__(self):
        self.action_one_hot = []
        self.observation_history = []
        self.reward_history = []
        self.done_history = []
        self.info_history = []

    @staticmethod
    def _to_storable(value: Any):
        if torch.is_tensor(value):
            return value.detach().cpu()

        if isinstance(value, dict):
            return {k: Observer._to_storable(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [Observer._to_storable(v) for v in value]

        return value

    def record_step(self, observation: Any, action_one_hot: Any, reward: float = 0.0, done: bool = False, info: dict | None = None):
        self.observation_history.append(self._to_storable(observation))
        self.action_one_hot.append(self._to_storable(action_one_hot))
        self.reward_history.append(float(reward))
        self.done_history.append(bool(done))
        self.info_history.append(self._to_storable(info) if info is not None else {})

    def clear(self):
        self.action_one_hot.clear()
        self.observation_history.clear()
        self.reward_history.clear()
        self.done_history.clear()
        self.info_history.clear()

    def to_episode_dict(self, metadata: dict | None = None):
        return {
            "version": 1,
            "metadata": metadata or {},
            "num_steps": len(self.observation_history),
            "observations": self.observation_history,
            "actions": self.action_one_hot,
            "rewards": self.reward_history,
            "dones": self.done_history,
            "infos": self.info_history,
        }

    def save_episode(self, file_path: str | Path, metadata: dict | None = None, clear_after_save: bool = True):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = self.to_episode_dict(metadata=metadata)
        torch.save(payload, file_path)

        if clear_after_save:
            self.clear()

    def append_episode(self, file_path: str | Path, metadata: dict | None = None, clear_after_save: bool = True):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        episode = self.to_episode_dict(metadata=metadata)
        if file_path.exists():
            try:
                data = torch.load(file_path, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(file_path, map_location="cpu")
        else:
            data = {
                "version": 1,
                "episodes": [],
            }

        if "episodes" not in data:
            data = {
                "version": 1,
                "episodes": [data],
            }

        data["episodes"].append(episode)
        torch.save(data, file_path)

        if clear_after_save:
            self.clear()

    @staticmethod
    def load(file_path: str | Path):
        file_path = Path(file_path)
        try:
            data = torch.load(file_path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(file_path, map_location="cpu")
        return data
