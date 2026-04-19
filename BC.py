import torch
from pathlib import Path
from torch.nn import CrossEntropyLoss
from staragent import StarAgent

class BehaviorCloningAgent(StarAgent):
    def __init__(
        self,
        dataset_path: str = "datasets/expert_zerg.pt",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self.dataset = self._load_dataset(self.dataset_path)
        self.loss_fn = CrossEntropyLoss()

    @staticmethod
    def _load_dataset(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        return data

    @staticmethod
    def _flatten_dataset(data: dict):
        if "episodes" in data:
            episodes = data["episodes"]
        else:
            episodes = [data]

        observations = []
        actions = []
        rewards = []
        for ep in episodes:
            observations.extend(ep.get("observations", []))
            actions.extend(ep.get("actions", []))
            rewards.extend(ep.get("rewards", []))

        return observations, actions, rewards

    @staticmethod
    def _to_tensor(x, dtype, device):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    @staticmethod
    def _action_to_index(action):
        if torch.is_tensor(action):
            flat = action.flatten()
            if flat.numel() == 1:
                return int(flat.item())
            return int(torch.argmax(flat).item())

        if isinstance(action, (list, tuple)):
            if len(action) == 1:
                return int(action[0])
            action_tensor = torch.as_tensor(action)
            return int(torch.argmax(action_tensor).item())

        return int(action)

    def learn(self, batch_size: int = 64, epochs: int = 5):
        observations, actions, rewards = self._flatten_dataset(self.dataset)

        dataset_size = len(observations)
        if dataset_size == 0:
            raise ValueError("Dataset is empty. Collect expert trajectories first.")

        if len(actions) != dataset_size:
            raise ValueError("Dataset observations/actions length mismatch")

        if len(rewards) != dataset_size:
            rewards = [0.0] * dataset_size

        indices = torch.arange(dataset_size)
        for epoch in range(epochs):
            permuted = indices[torch.randperm(dataset_size)]
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = permuted[start:end].tolist()
                batch_obs = [observations[i] for i in batch_indices]
                batch_actions = torch.tensor(
                    [self._action_to_index(actions[i]) for i in batch_indices],
                    dtype=torch.long,
                    device=self.device,
                )
                batch_rewards = torch.tensor(
                    [float(rewards[i]) for i in batch_indices],
                    dtype=torch.float32,
                    device=self.device,
                )

                image_tensors = torch.stack(
                    [self._to_tensor(obs["image"], dtype=torch.float32, device=self.device) for obs in batch_obs]
                )
                resource_tensors = torch.stack(
                    [self._to_tensor(obs["resources"], dtype=torch.float32, device=self.device) for obs in batch_obs]
                )
                self._reset_lstm_state(batch_size=image_tensors.size(0))
                policy_inputs = self._encode_temporal_features(image_tensors, resource_tensors)
                logits = self.actioner(policy_inputs)
                values = self.critic(policy_inputs).squeeze(1)
                advantages = batch_rewards - values.detach()
                action_loss = self.loss_fn(logits, batch_actions)
                value_loss = advantages.pow(2).mean()
                total_loss = action_loss + 0.5 * value_loss

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.feature_extractor.parameters())
                    + list(self.actioner.parameters())
                    + list(self.critic.parameters())
                    + list(self.lstm.parameters()),
                    max_norm=2.0,
                )
                self.optimizer.step()
            print(f"[BC] epoch={epoch + 1}/{epochs} action_loss={action_loss.item():.4f} value_loss={value_loss.item():.4f}")
        self._save_checkpoint(Path("checkpoints/bc_agent.pt"))