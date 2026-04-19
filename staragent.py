from scaffold import Scaffold
from pathlib import Path
from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId
from utils import MLP, ImageFeatureExtractor
import torch
from torch.distributions import Categorical
from torch.nn import LSTM


class StarAgent(Scaffold):
    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_lstm_layers: int = 2,
        out_channels: int = 64,
        gamma: float = 0.99,
        lr: float = 3e-4,
        entropy_coef: float = 0.01,
        train_mode: bool = True,
        action_interval: int = 8,
        rollout_horizon: int = 64,
        checkpoint_path: str = "checkpoints/staragent_policy.pt",
        device: str | torch.device | None = None,
    ):
        super().__init__()

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.train_mode = train_mode
        self.action_interval = max(1, action_interval)
        self.rollout_horizon = max(1, int(rollout_horizon))
        self.checkpoint_path = Path(checkpoint_path)
        self.device = self._resolve_device(device)
        self.hidden_dim = int(hidden_dim)
        self.n_lstm_layers = max(1, int(n_lstm_layers))
        self.total_actions = 30

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
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters())
            + list(self.actioner.parameters())
            + list(self.critic.parameters())
            + list(self.lstm.parameters()),
            lr=lr,
        )

        if self.train_mode:
            self.feature_extractor.train()
            self.actioner.train()
            self.critic.train()
            self.lstm.train()
        else:
            self.feature_extractor.eval()
            self.actioner.eval()
            self.critic.eval()
            self.lstm.eval()

        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_logits = []
        self.episode_counter = 0
        self.lstm_state = None

        print(f"[StarAgent] Using device: {self.device}")
        self._load_checkpoint()

    async def _execute_action(self, action_idx: int):
        # See Scaffold for action mapping
        if action_idx == 0:
            return False
        if action_idx == 1:
            return await self.build_spawning_pool()
        if action_idx == 2:
            return await self.train_zerglings(6)
        if action_idx == 3:
            return await self.attack_move(target=self._get_attack_target())
        if action_idx == 4:
            return await self.train_drones(1)
        if action_idx == 5:
            return await self.train_overlord(1)
        if action_idx == 6:
            return await self.train_anti_air(2)
        if action_idx == 7:
            return await self.train_flying_unit(2)
        if action_idx == 8:
            return await self.build_hydralisk_den()
        if action_idx == 9:
            return await self.build_spire()
        if action_idx == 10:
            return await self.build_roach_warren()
        if action_idx == 11:
            return await self.train_roach(2)
        if action_idx == 12:
            return await self.build_baneling_nest()
        if action_idx == 13:
            return await self.train_baneling(2)
        if action_idx == 14:
            return await self.build_infestation_pit()
        if action_idx == 15:
            return await self.build_greater_spire()
        if action_idx == 16:
            return await self.train_brood_lord(1)
        if action_idx == 17:
            return await self.build_spine_crawler()
        if action_idx == 18:
            return await self.build_spore_crawler()
        if action_idx == 19:
            return await self.inject_larva()
        if action_idx == 20:
            return await self.spread_creep()
        if action_idx == 21:
            return await self.transfuse()
        if action_idx == 22:
            return await self.research_zergling_speed()
        if action_idx == 23:
            return await self.research_burrow()
        if action_idx == 24:
            return await self.research_roach_speed()
        if action_idx == 25:
            return await self.research_baneling_speed()
        if action_idx == 26:
            return await self.research_flyer_attacks()
        if action_idx == 27:
            return await self.retreat()
        if action_idx == 28:
            return await self.regroup()
        if action_idx == 29:
            return await self.focus_fire()
        return False

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        requested = str(device).strip()
        if requested.lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            resolved = torch.device(requested)
        except (RuntimeError, TypeError, ValueError):
            print(f"[StarAgent] Invalid device '{requested}', falling back to cpu")
            return torch.device("cpu")

        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                print("[StarAgent] CUDA requested but unavailable, falling back to cpu")
                return torch.device("cpu")

            if (
                resolved.index is not None
                and resolved.index >= torch.cuda.device_count()
            ):
                print(
                    f"[StarAgent] CUDA device index {resolved.index} unavailable, falling back to cuda:0"
                )
                return torch.device("cuda:0")

        return resolved

    def _load_checkpoint(self):
        if not self.checkpoint_path.exists():
            return
        try:
            try:
                state = torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                state = torch.load(self.checkpoint_path, map_location=self.device)
            self.feature_extractor.load_state_dict(state["feature_extractor"])
            self.actioner.load_state_dict(state["actioner"])
            if "lstm" in state:
                self.lstm.load_state_dict(state["lstm"])
            else:
                print("[StarAgent] No LSTM in checkpoint; initializing fresh")
            if "critic" in state:
                self.critic.load_state_dict(state["critic"])
            else:
                print("[StarAgent] No critic in checkpoint; initializing fresh")
            if "optimizer" in state and self.train_mode:
                self.optimizer.load_state_dict(state["optimizer"])
            self.episode_counter = int(state.get("episode_counter", 0))
            print(f"[StarAgent] Loaded checkpoint from {self.checkpoint_path}")
        except Exception as exc:
            print(f"[StarAgent] Failed to load checkpoint ({exc}); starting fresh")

    def save_checkpoint(self, path=None):
        if path is None:
            path = self.checkpoint_path
        if type(path) == str:
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "feature_extractor": self.feature_extractor.state_dict(),
            "actioner": self.actioner.state_dict(),
            "critic": self.critic.state_dict(),
            "lstm": self.lstm.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episode_counter": self.episode_counter,
        }
        torch.save(state, path)

    def _reset_lstm_state(self, batch_size: int = 1):
        h0 = torch.zeros(
            self.n_lstm_layers,
            batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float32,
        )
        c0 = torch.zeros(
            self.n_lstm_layers,
            batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.lstm_state = (h0, c0)

    def _encode_temporal_features(self, image_tensor, resource_tensor):
        spatial_features = self.feature_extractor(image_tensor)
        lstm_input = torch.cat([spatial_features, resource_tensor], dim=1).unsqueeze(1)

        if self.lstm_state is None:
            self._reset_lstm_state(batch_size=lstm_input.size(0))

        lstm_out, new_state = self.lstm(lstm_input, self.lstm_state)
        # Truncate history to keep memory bounded across long episodes.
        self.lstm_state = (new_state[0].detach(), new_state[1].detach())
        return lstm_out[:, -1, :]

    def _observation_tensors(self):
        image_obs, resource_obs = self.get_observation()
        image_tensor = (
            torch.from_numpy(image_obs)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float32)
        )
        resource_tensor = (
            torch.from_numpy(resource_obs)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float32)
        )
        return image_tensor, resource_tensor

    async def _execute_action(self, action_idx: int):
        if action_idx == 0:
            return False
        if action_idx == 1:
            return await self.train_drones(1)
        if action_idx == 2:
            return await self.train_overlord(1)
        if action_idx == 3:
            return await self.build_extractor()
        if action_idx == 4:
            return await self.build_spawning_pool()
        if action_idx == 5:
            return await self.gather_minerals()
        if action_idx == 6:
            return await self.gather_vespene()
        if action_idx == 7:
            return await self.train_zerglings(2)
        if action_idx == 8:
            return await self.attack_move(target=self.enemy_start_locations[0])
        if action_idx == 9:
            return await self.rally_army()
        if action_idx == 10:
            return await self.build_baneling_nest()
        if action_idx == 11:
            return await self.train_banelings(2)
        if action_idx == 12:
            return await self.expand()
        return False

    def _finish_episode(self, terminal_reward: float):
        if not self.episode_log_probs:
            return

        self.episode_rewards[-1] += terminal_reward

        discounted = []
        running = 0.0
        for reward in reversed(self.episode_rewards):
            running = reward + self.gamma * running
            discounted.append(running)
        discounted.reverse()

        returns = torch.tensor(discounted, dtype=torch.float32, device=self.device)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        image_tensor, resource_tensor = self._observation_tensors()
        value_input = self._encode_temporal_features(image_tensor, resource_tensor)
        state_value = self.critic(value_input).squeeze(1)
        advantage = returns - state_value
        critic_loss = advantage.pow(2).mean()

        log_probs = torch.stack(self.episode_log_probs)
        rl_loss = -(log_probs * advantage.detach()).sum()

        logits = torch.stack(self.episode_logits)
        dist = Categorical(logits=logits)
        entropy_loss = dist.entropy().mean()

        total_loss = rl_loss + 0.5 * critic_loss - self.entropy_coef * entropy_loss

        if self.train_mode:
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

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self._log_entropy(entropy_loss.item())
            self._log_train(total_loss.item(), rl_loss.item(), critic_loss.item())

        self.episode_log_probs.clear()
        self.episode_rewards.clear()
        self.episode_logits.clear()

    async def on_start(self):
        self._prev_metrics = self._collect_metrics()
        self._reset_lstm_state()

    async def on_step(self, iteration: int):
        if self.workers.idle and self.mineral_field:
            for worker in self.workers.idle[:2]:
                self.do(worker.gather(self.mineral_field.closest_to(worker)))

        if iteration % self.action_interval != 0:
            return

        image_tensor, resource_tensor = self._observation_tensors()
        if self.train_mode:
            policy_input = self._encode_temporal_features(image_tensor, resource_tensor)
            logits = self.actioner(policy_input)

            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).squeeze(0)
            action_idx = int(action.item())
        else:
            with torch.no_grad():
                policy_input = self._encode_temporal_features(
                    image_tensor, resource_tensor
                )
                logits = self.actioner(policy_input)
                dist = Categorical(logits=logits)
                action_idx = int(dist.sample().item())
            log_prob = None

        action_ok = await self._execute_action(action_idx)
        reward = self._compute_step_reward(action_ok)

        self._log_step(iteration, action_idx, action_ok, reward)

        if self.train_mode:
            self.episode_log_probs.append(log_prob)
            self.episode_logits.append(logits.detach())
            self.episode_rewards.append(reward)

            if len(self.episode_log_probs) >= self.rollout_horizon:
                self._finish_episode(terminal_reward=0.0)

    async def on_end(self, result: Result):
        self.episode_counter += 1

        if result == Result.Victory:
            terminal_reward = 2.0
        elif result == Result.Defeat:
            terminal_reward = -2.0
        else:
            terminal_reward = -0.25

        if self.episode_rewards:
            self._finish_episode(terminal_reward=terminal_reward)

        metrics = self._collect_metrics()
        self._log_episode_end(result, metrics)
        self.lstm_state = None
