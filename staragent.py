from scaffold import Scaffold
from pathlib import Path
from sc2.data import Result
from utils import MLP, ImageFeatureExtractor
import torch
from torch.distributions import Categorical
from torch.nn import LSTM
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import mlflow
import numpy as np

class StarAgent(Scaffold):
    """PPO Zerg agent with CNN + LSTM + Transformer.

    Uses full 43 actions from Scaffold for maximum strategy.
    """

    AGENT_TOTAL_ACTIONS = None  # Use scaffold's 43 actions

    def __init__(
        self,
        name: str = "StarAgent",
        log_mlflow: bool = False,
        hidden_dim: int = 256,
        n_layers: int = 5,
        n_lstm_layers: int = 1,
        out_channels: int = 64,
        gamma: float = 0.99,
        lr: float = 3e-4,
        lr_decay: float = 0.95,
        lr_decay_steps: int = 50,
        entropy_coef: float = 0.01,
        ppo_clip_eps: float = 0.2,
        grad_clip_norm: float = 1.0,
        train_mode: bool = True,
        action_interval: int = 8,
        rollout_horizon: int = 128,
        use_attention: bool = True,
        device: str | torch.device | None = None,
    ):
        super().__init__()

        self.name = name
        self.log_mlflow = log_mlflow
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.ppo_clip_eps = ppo_clip_eps
        self.grad_clip_norm = float(grad_clip_norm)
        self.action_interval = max(1, action_interval)
        self.rollout_horizon = max(1, int(rollout_horizon))
        self.use_attention = use_attention
        self.device = self._resolve_device(device)
        self.hidden_dim = int(hidden_dim)
        self.n_lstm_layers = max(1, int(n_lstm_layers))
        self.out_channels = int(out_channels)
        self.lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps
        self.initial_lr = lr

        image_channels = self.observation_space[0][0]
        self.feature_extractor = ImageFeatureExtractor(
            in_channels=image_channels,
            out_channels=out_channels,
        ).to(self.device)
        self.actioner = MLP(
            input_dim=self.hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=self.total_actions,
            n_layers=n_layers,
            lr=lr,
        ).to(self.device)
        self.critic = MLP(
            input_dim=self.hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_layers=n_layers,
            lr=lr,
        ).to(self.device)
        self.lstm = LSTM(
            input_size=out_channels + self.observation_space[1],
            hidden_size=self.hidden_dim,
            num_layers=self.n_lstm_layers,
            batch_first=True,
        ).to(self.device)

        if use_attention:
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
            )
            self.transformer = TransformerEncoder(encoder_layer, num_layers=2).to(self.device)
        else:
            self.transformer = None

        self.optimizer = torch.optim.Adam(
            self._all_parameters(),
            lr=lr,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_decay_steps,
            gamma=lr_decay,
        )

        # Set train/eval mode via the property so models stay in sync.
        self._train_mode: bool = False          # backing field
        self.train_mode = train_mode            # triggers property setter

        self.episode_log_probs: list[torch.Tensor] = []
        self.episode_rewards: list[float] = []
        self.episode_values: list[torch.Tensor] = []   # kept for backwards compatibility (not used)
        self.episode_logits: list[torch.Tensor] = []
        # Rollout buffers (store on host to save GPU memory)
        self.rollout_images: list[np.ndarray] = []
        self.rollout_resources: list[np.ndarray] = []
        self.episode_counter = 0
        self.lstm_state: tuple | None = None
        self.episode_actions: list[int] = []

        print(f"[StarAgent] Using device: {self.device}")

        if self.log_mlflow:
            mlflow.log_param(f"{self.name}/hidden_dim", hidden_dim)
            mlflow.log_param(f"{self.name}/n_layers", n_layers)
            mlflow.log_param(f"{self.name}/n_lstm_layers", n_lstm_layers)
            mlflow.log_param(f"{self.name}/out_channels", out_channels)
            mlflow.log_param(f"{self.name}/gamma", gamma)
            mlflow.log_param(f"{self.name}/lr", lr)
            mlflow.log_param(f"{self.name}/entropy_coef", entropy_coef)
            mlflow.log_param(f"{self.name}/grad_clip_norm", grad_clip_norm)
            mlflow.log_param(f"{self.name}/train_mode", train_mode)
            mlflow.log_param(f"{self.name}/action_interval", action_interval)
            mlflow.log_param(f"{self.name}/rollout_horizon", rollout_horizon)
            mlflow.log_param(f"{self.name}/device", str(self.device))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def train_mode(self) -> bool:
        return self._train_mode

    @train_mode.setter
    def train_mode(self, value: bool):
        self._train_mode = bool(value)
        mode_fn = "train" if self._train_mode else "eval"
        for model in (self.feature_extractor, self.actioner, self.critic, self.lstm, self.transformer):
            if model is not None:
                getattr(model, mode_fn)()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _all_parameters(self) -> list:
        """Single source of truth for all trainable parameters."""
        params = (
            list(self.feature_extractor.parameters())
            + list(self.actioner.parameters())
            + list(self.critic.parameters())
            + list(self.lstm.parameters())
        )
        if self.transformer is not None:
            params += list(self.transformer.parameters())
        return params

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None or str(device).strip().lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        requested = str(device).strip()
        try:
            resolved = torch.device(requested)
        except (RuntimeError, TypeError, ValueError):
            print(f"[StarAgent] Invalid device '{requested}', falling back to cpu")
            return torch.device("cpu")

        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                print("[StarAgent] CUDA requested but unavailable, falling back to cpu")
                return torch.device("cpu")
            if resolved.index is not None and resolved.index >= torch.cuda.device_count():
                print(
                    f"[StarAgent] CUDA device index {resolved.index} unavailable,"
                    " falling back to cuda:0"
                )
                return torch.device("cuda:0")

        return resolved

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def load_checkpoint(self, path):
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            return
        try:
            try:
                state = torch.load(
                    path,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                state = torch.load(path, map_location=self.device)

            self.feature_extractor.load_state_dict(state["feature_extractor"])
            self.actioner.load_state_dict(state["actioner"])

            for key, model, name in [
                ("lstm",      self.lstm,      "LSTM"),
                ("critic",    self.critic,    "critic"),
            ]:
                if key in state:
                    model.load_state_dict(state[key])
                else:
                    print(f"[StarAgent] No {name} in checkpoint; initializing fresh")

            if "optimizer" in state and self.train_mode:
                self.optimizer.load_state_dict(state["optimizer"])

            self.episode_counter = int(state.get("episode_counter", 0))
            print(f"[StarAgent] Loaded checkpoint from {path}")
        except Exception as exc:
            print(f"[StarAgent] Failed to load checkpoint ({exc}); starting fresh")

    def save_checkpoint(self, path):
        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "feature_extractor": self.feature_extractor.state_dict(),
            "actioner":          self.actioner.state_dict(),
            "critic":            self.critic.state_dict(),
            "lstm":              self.lstm.state_dict(),
            "optimizer":         self.optimizer.state_dict(),
            "episode_counter":   self.episode_counter,
        }
        torch.save(state, path)

    # ------------------------------------------------------------------
    # LSTM / observation helpers
    # ------------------------------------------------------------------

    def _reset_lstm_state(self, batch_size: int = 1):
        zeros = lambda: torch.zeros(
            self.n_lstm_layers, batch_size, self.hidden_dim,
            device=self.device, dtype=torch.float32,
        )
        self.lstm_state = (zeros(), zeros())

    def _encode_temporal_features(self, image_tensor, resource_tensor):
        spatial_features = self.feature_extractor(image_tensor)
        lstm_input = torch.cat([spatial_features, resource_tensor], dim=1).unsqueeze(1)

        if self.lstm_state is None:
            self._reset_lstm_state(batch_size=lstm_input.size(0))

        lstm_out, new_state = self.lstm(lstm_input, self.lstm_state)
        # Detach to bound memory across rollouts without breaking the current graph.
        self.lstm_state = (new_state[0].detach(), new_state[1].detach())
        return lstm_out[:, -1, :]

    def _observation_tensors(self):
        image_obs, resource_obs = self.get_observation()
        image_tensor = (
            torch.from_numpy(image_obs).unsqueeze(0).to(self.device, dtype=torch.float32)
        )
        resource_tensor = (
            torch.from_numpy(resource_obs).unsqueeze(0).to(self.device, dtype=torch.float32)
        )
        return image_tensor, resource_tensor

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _finish_episode(self, terminal_reward: float):
        """Compute PPO loss over the stored rollout and update weights.

        This implementation keeps most stored rollout data on the host (CPU) and
        recomputes logits/values in a single batched forward pass on the device
        to avoid holding per-step autograd graphs on the GPU.
        """
        if not self.episode_rewards:
            return

        # Add terminal reward to last transition
        self.episode_rewards[-1] += terminal_reward

        T = len(self.episode_rewards)

        # Compute discounted returns on device
        discounted = []
        running = 0.0
        for r in reversed(self.episode_rewards):
            running = r + self.gamma * running
            discounted.append(running)
        discounted.reverse()
        returns = torch.tensor(discounted, dtype=torch.float32, device=self.device)

        # Prepare actions tensor
        actions_tensor = torch.tensor(self.episode_actions, dtype=torch.long, device=self.device)

        # Prepare old log-probs: they were stored on CPU (possibly pinned), move to device
        if self.episode_log_probs:
            old_log_probs = torch.stack(self.episode_log_probs)
            if old_log_probs.dim() > 1:
                old_log_probs = old_log_probs.squeeze(-1)
            old_log_probs = old_log_probs.to(self.device, non_blocking=True)
        else:
            old_log_probs = torch.zeros(T, device=self.device)

        # Recompute logits and values for the whole rollout in a single batched pass
        images_np = np.stack(self.rollout_images, axis=0).astype(np.float32)
        resources_np = np.stack(self.rollout_resources, axis=0).astype(np.float32)

        if self.device.type == "cuda":
            images_tensor = torch.from_numpy(images_np).pin_memory().to(self.device, non_blocking=True)
            resources_tensor = torch.from_numpy(resources_np).pin_memory().to(self.device, non_blocking=True)
        else:
            images_tensor = torch.from_numpy(images_np)
            resources_tensor = torch.from_numpy(resources_np)

        spatial_features = self.feature_extractor(images_tensor)
        lstm_inputs = torch.cat([spatial_features, resources_tensor], dim=1).unsqueeze(0)

        # Run LSTM over the full sequence (batch_size=1, seq_len=T)
        h0 = torch.zeros(self.n_lstm_layers, 1, self.hidden_dim, device=self.device, dtype=torch.float32)
        c0 = torch.zeros(self.n_lstm_layers, 1, self.hidden_dim, device=self.device, dtype=torch.float32)
        lstm_out, _ = self.lstm(lstm_inputs, (h0, c0))
        lstm_out = lstm_out.squeeze(0)

        logits_batch = self.actioner(lstm_out)
        values_batch = self.critic(lstm_out).squeeze(-1)

        dist = Categorical(logits=logits_batch)
        new_log_probs = dist.log_prob(actions_tensor)

        # Advantage estimation — use stored per-step values (from the acting policy)
        if self.episode_values:
            old_values = torch.tensor(self.episode_values, dtype=values_batch.dtype, device=self.device)
            advantage = returns - old_values.detach()
        else:
            advantage = returns - values_batch.detach()
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

        # Losses
        critic_loss = (returns - values_batch).pow(2).mean()
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip_eps, 1 + self.ppo_clip_eps) * advantage
        ppo_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = dist.entropy().mean()

        total_loss = ppo_loss + 0.5 * critic_loss - self.entropy_coef * entropy_loss

        # Update
        if self.train_mode:
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._all_parameters(), self.grad_clip_norm)
            self.optimizer.step()
            self.lr_scheduler.step()

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        current_lr = self.optimizer.param_groups[0]["lr"]
        self._log_entropy(entropy_loss.item())
        self._log_train(total_loss.item(), ppo_loss.item(), critic_loss.item())

        # MLflow metrics (log before clearing buffers)
        total_episode_reward = float(sum(self.episode_rewards))
        if self.log_mlflow:
            mlflow.log_metric(f"{self.name}/train/total_loss", total_loss.item(), step=self.episode_counter)
            mlflow.log_metric(f"{self.name}/train/ppo_loss", ppo_loss.item(), step=self.episode_counter)
            mlflow.log_metric(f"{self.name}/train/critic_loss", critic_loss.item(), step=self.episode_counter)
            mlflow.log_metric(f"{self.name}/train/entropy", entropy_loss.item(), step=self.episode_counter)
            mlflow.log_metric(f"{self.name}/train/episode_reward", total_episode_reward, step=self.episode_counter)
            mlflow.log_metric(f"{self.name}/train/learning_rate", current_lr, step=self.episode_counter)

        # Clear host buffers
        self.episode_log_probs.clear()
        self.episode_rewards.clear()
        self.episode_values.clear()
        self.rollout_images.clear()
        self.rollout_resources.clear()
        self.episode_actions.clear()

    # ------------------------------------------------------------------
    # SC2 lifecycle hooks
    # ------------------------------------------------------------------

    async def on_start(self):
        self._prev_metrics = self._collect_metrics()
        self._reset_lstm_state()

    async def on_step(self, iteration: int):
        # Always keep idle workers busy.
        if self.workers.idle and self.mineral_field:
            for worker in self.workers.idle[:2]:
                self.do(worker.gather(self.mineral_field.closest_to(worker)))

        if iteration % self.action_interval != 0:
            return

        # Use raw numpy observations so we can store them on host (CPU)
        image_obs, resource_obs = self.get_observation()

        # Convert for the immediate forward pass
        image_tensor = torch.from_numpy(image_obs).unsqueeze(0).to(self.device, dtype=torch.float32)
        resource_tensor = torch.from_numpy(resource_obs).unsqueeze(0).to(self.device, dtype=torch.float32)

        if self.train_mode:
            # Full forward pass — keep graph only for this immediate computation.
            policy_input = self._encode_temporal_features(image_tensor, resource_tensor)
            logits = self.actioner(policy_input)
            value = self.critic(policy_input)

            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).squeeze(0)
            action_idx = int(action.item())
        else:
            with torch.no_grad():
                policy_input = self._encode_temporal_features(image_tensor, resource_tensor)
                logits = self.actioner(policy_input)
                value = self.critic(policy_input)
                dist = Categorical(logits=logits)
                action_idx = int(dist.sample().item())
            log_prob = None

        action_ok = await self._execute_action(action_idx)
        reward = self._compute_step_reward(action_ok)
        self._log_step(iteration, action_idx, action_ok, reward)

        if self.log_mlflow:
            mlflow.log_metric(f"{self.name}/step/reward", reward, step=iteration + self.episode_counter * 1000)
            mlflow.log_metric(f"{self.name}/step/action_success", int(action_ok if action_ok is not None else 0), step=iteration + self.episode_counter * 1000)

        if self.train_mode:
            # Store raw observations on host so we can recompute model outputs in a batched way later.
            self.rollout_images.append(np.array(image_obs, copy=True))
            self.rollout_resources.append(np.array(resource_obs, copy=True))

            # Store old log-prob on CPU (pinned if using CUDA for faster transfer)
            if log_prob is not None:
                if self.device.type == "cuda":
                    self.episode_log_probs.append(log_prob.detach().cpu().pin_memory())
                else:
                    self.episode_log_probs.append(log_prob.detach().cpu())

            # Store critic value as a small CPU scalar for correct advantage estimation
            try:
                self.episode_values.append(float(value.detach().cpu().item()))
            except Exception:
                self.episode_values.append(0.0)

            self.episode_rewards.append(reward)
            self.episode_actions.append(action_idx)

            # If we've collected enough steps, run the PPO update.
            if len(self.rollout_images) >= self.rollout_horizon:
                self._finish_episode(terminal_reward=0.0)

    async def on_end(self, result: Result):
        self.episode_counter += 1

        terminal_reward = {
            Result.Victory: 5.0,
            Result.Defeat:  -5.0,
        }.get(result, -2.5)

        if self.episode_rewards:
            self._finish_episode(terminal_reward=terminal_reward)

        self._log_episode_end(result, self._collect_metrics())

        if self.log_mlflow:
            mlflow.log_metric(f"{self.name}/episode/result", 1.0 if result == Result.Victory else 0.0, step=self.episode_counter)
            mlflow.log_metric(f"{self.name}/episode/terminal_reward", terminal_reward, step=self.episode_counter)

        self.lstm_state = None