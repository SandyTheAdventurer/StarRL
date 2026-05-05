from scaffold import Scaffold, N_ACTION_PARAMS, ACTION_PARAM_RANGES
import torch
from itertools import chain
from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from utils import MLP, ImageFeatureExtractor, DiscreteHead, EntityEncoder, RolloutBuffer, RunningMeanStd, CrossAttentionFusion
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from line_profiler import profile
except Exception:
    def profile(func):
        return func

try:
    import mlflow
except Exception:
    mlflow = None


class ParameterHead(nn.Module):
    """Outputs N_ACTION_PARAMS continuous values in (0, 1) via sigmoid.

    These modulate how actions are executed:
      param[0] = quantity    – how many units/orders to issue
      param[1] = aggression  – military commitment level
      param[2] = location_x   – for attack/retreat to location: normalized x coord 0-1
      param[3] = location_y   – for attack/retreat to location: normalized y coord 0-1
      param[4,5] = reserved for future use

    Trained jointly with the policy via a Gaussian log-probability loss
    so the model learns to pair the right parameters with each action.
    """
    def __init__(self, input_dim: int, n_params: int = N_ACTION_PARAMS, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_params),
        )
        self.log_std = nn.Parameter(torch.zeros(n_params))  # learnable std per param

    def forward(self, x):
        mean = torch.sigmoid(self.net(x))          # (B, n_params) in (0,1)
        std  = torch.exp(self.log_std.clamp(-4, 1)) # (n_params,)
        return mean, std

    def sample(self, x):
        """Sample params and return (params, log_prob)."""
        mean, std = self(x)
        dist = torch.distributions.Normal(mean, std)
        raw  = dist.rsample()                      # reparameterised
        params = torch.sigmoid(raw)                # clamp to (0,1)
        # log_prob with change-of-variables for sigmoid squashing
        log_prob = dist.log_prob(raw) - torch.log(params * (1 - params) + 1e-6)
        return params, log_prob.sum(dim=-1)        # sum over param dims

class StarAgent(Scaffold):
    def __init__(self,
                name = "StarAgent",
                lr = 3e-4,
                n_layers = 6,
                hidden_channels = 64,
                hidden_size = 512,
                n_lstm_layers = 2,
                n_transformer_layers = 1,
                n_critic_layers = 4,
                critic_hidden_size = 512,
                n_heads = 2,
                transformer_hidden_size = 64,
                lstm_hidden_size = 128,
                rollout_size = 2048,
                minibatch_size = 256,
                seq_len = 64,
                max_units_tracked = 64,
                reward_weights = None,
                train_mode = True,
                use_critic: bool = True,
                min_attack_supply: float = 4.0,
                min_attack_units: int = 4,
                attack_defend_radius: float = 30.0,
                phase: int = 3,
                normalize_returns = True,
                log_mlflow = True,
                device = 'cuda' if torch.cuda.is_available() else 'cpu',
                buffer_device = 'cuda' if torch.cuda.is_available() else 'cpu',
                compile_model: bool = False,
                compile_backend: str = "inductor",
                compile_mode: str = "default",
                compile_dynamic: bool = True,
                compile_on_cpu: bool = False,
                ent_coef: float = 0.05,
                vf_coef: float = 0.5,
                clip_range_vf: float | None = 0.2,
                normalize_advantage: bool = True,
                target_kl: float | None = None,
                gamma: float = 0.99,
                lam: float = 0.95,
                clip_range: float = 0.2,
                max_grad_norm: float = 0.5,
                lr_schedule: str | None = None,
                lr_warmup_steps: int = 0,
                entropy_target: float | None = None,
                param_supervised_coef: float = 0.1,
                ppo_epochs: int = 3,
                log_level = 0):
        super().__init__(
            max_units_tracked,
            reward_weights,
            log_level,
            min_attack_supply=min_attack_supply,
            min_attack_units=min_attack_units,
            attack_defend_radius=attack_defend_radius,
            phase=phase,
        )

        self.name = name
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.n_transformer_layers = n_transformer_layers
        self.n_heads = n_heads
        self.transformer_hidden_size = transformer_hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.fusion_dim = hidden_channels + hidden_size // 2 + transformer_hidden_size
        self.device = device
        print(f"Using device: {self.device}")
        self.use_amp = self.device.startswith("cuda") and torch.cuda.is_available()
        self.buffer_device = buffer_device
        self.log_mlflow = log_mlflow and mlflow is not None
        self.compile_model = bool(compile_model)
        self.compile_backend = compile_backend
        self.compile_mode = compile_mode
        self.compile_dynamic = compile_dynamic
        self.compile_on_cpu = compile_on_cpu
        self._compile_warned = False
        self._mlflow_step = 0
        self._episode_reward_sum = 0.0
        self._episode_reward_count = 0
        self.scaler = GradScaler(self.device, enabled=self.use_amp)
        self._ppo_updates_total = 0
        self._ppo_updates_in_episode = 0
        self.use_critic = bool(use_critic)
        self.normalize_returns = normalize_returns
        self.return_rms = RunningMeanStd()
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        self.ppo_update_epochs = ppo_epochs
        self.lr_schedule = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.entropy_target = entropy_target
        self._use_return_normalization = normalize_returns
        self.ent_coef = ent_coef
        self._ent_coef_adaptive = ent_coef
        self.vf_coef = vf_coef
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.param_supervised_coef = param_supervised_coef

        self.rollout_buffer = RolloutBuffer(device=buffer_device)
        self.rollout_buffer.reset()
        self.rollout_size = rollout_size
        self.minibatch_size = minibatch_size
        self.seq_len = seq_len
        assert self.seq_len <= self.minibatch_size, "seq_len must be less than or equal to minibatch_size"
        assert self.rollout_size % self.minibatch_size == 0, "rollout_size must be divisible by minibatch_size"
        self.fusion_attn = CrossAttentionFusion(
            image_dim=hidden_channels,
            resource_dim=hidden_size // 2,
            entity_dim=transformer_hidden_size,
            fusion_dim=self.fusion_dim,
            n_heads=n_heads,
        )
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)

        self.image_extractor = ImageFeatureExtractor(
                in_channels=self.observation_space[0][0],
                out_channels=hidden_channels,
        )

        self.resource_extractor = MLP(
                input_dim=self.observation_space[1],
                hidden_dim=hidden_size // 2,
                output_dim=hidden_size // 2,
                n_layers=n_layers,
        )

        entity_feature_dim = 11
        self.entity_extractor = EntityEncoder(
            input_dim=entity_feature_dim,
                d_model=transformer_hidden_size,
                n_heads=n_heads,
                n_layers=n_transformer_layers,
        )

        self.lstm = nn.LSTM(
            input_size=self.fusion_dim,
                hidden_size=lstm_hidden_size,
                num_layers=n_lstm_layers,
                batch_first=True,
        )

        self.mlp = MLP(
                input_dim=lstm_hidden_size,
                hidden_dim=hidden_size,
                output_dim=hidden_size,
                n_layers=n_layers,
        )

        self.critic = MLP(
                input_dim=lstm_hidden_size,
                hidden_dim=critic_hidden_size,
                output_dim=1,
                n_layers=n_critic_layers,
                layer_norm=True,
                init_orthogonal=True,
        )

        self.action_head = DiscreteHead(
                input_dim=hidden_size,
                output_dim=self.total_actions
        )
        self.param_head = ParameterHead(
            input_dim=hidden_size,
            n_params=N_ACTION_PARAMS,
            hidden_dim=128,
        )
        self.train_mode = train_mode
        modules = [
            self.image_extractor,
            self.resource_extractor,
            self.entity_extractor,
            self.fusion_attn,
            self.fusion_norm,
            self.lstm,
            self.mlp,
            self.action_head,
            self.param_head,
        ]
        if self.use_critic:
            modules.insert(-1, self.critic)
        for module in modules:
            module.to(self.device)
            module.train(self.train_mode)
        if not self.use_critic:
            self.critic.to(self.device)
            self.critic.eval()
        self._maybe_compile_modules()
        self.params = list(chain(
            self.image_extractor.parameters(),
            self.resource_extractor.parameters(),
            self.entity_extractor.parameters(),
            self.fusion_attn.parameters(),
            self.fusion_norm.parameters(),
            self.lstm.parameters(),
            self.mlp.parameters(),
            self.action_head.parameters(),
            self.param_head.parameters(),
        ))
        if self.use_critic:
            self.params.extend(self.critic.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
        self._base_lr = lr
        if self.lr_schedule is not None:
            if self.lr_schedule == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=2000, eta_min=lr / 10
                )
            elif self.lr_schedule == "linear":
                self.scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=500
                )
            elif self.lr_schedule == "constant":
                self.scheduler = None
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=2000, eta_min=lr / 10
                )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=2000, eta_min=lr / 10
            )
        self._warmup_completed = False
        self.round_number = 0

    def set_round_number(self, round_num: int):
        """Set the current training round number."""
        self.round_number = round_num

    @staticmethod
    def gae(rewards, values, dones, bootstrap_value=None, gamma=0.99, lam=0.95):
        if bootstrap_value is None:
            bootstrap_value = torch.tensor(0.0, device=values.device, dtype=values.dtype)
        else:
            bootstrap_value = torch.as_tensor(bootstrap_value, device=values.device, dtype=values.dtype)

        values_next = torch.cat([values, bootstrap_value.view(1)])
        advantages = torch.zeros_like(rewards)
        last_advantage = 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            next_value = values_next[t + 1]

            delta = rewards[t] + gamma * next_value * mask - values[t]
            last_advantage = delta + gamma * lam * last_advantage * mask

            advantages[t] = last_advantage

        returns = advantages + values
        return advantages, returns

    def _sample_valid_starts(self, T: int, seq_len: int, dones, n_sequences: int):
        done_indices = set(torch.where(dones > 0)[0].tolist())
        
        done_arr = np.zeros(T, dtype=np.int32)
        for d in done_indices:
            if d < T:
                done_arr[d] = 1
        prefix = np.cumsum(done_arr)
        
        def has_done_in_range(i):
            end = i + seq_len - 2
            if end >= T:
                return True
            if i == 0:
                return prefix[end] > 0
            return (prefix[end] - prefix[i - 1]) > 0
        
        candidates = [i for i in range(T - seq_len + 1) if not has_done_in_range(i)]
        
        if not candidates:
            return [0]
        
        return np.random.choice(candidates, size=n_sequences, replace=True).tolist()
    
    def _reset_lstm_state(self, batch_size: int = 1):
        zeros = lambda: torch.zeros(
            self.n_lstm_layers, batch_size, self.lstm_hidden_size,
            device=self.device, dtype=torch.float32,
        )
        self.lstm_state = (zeros(), zeros())

    def _compile_module(self, module, name: str):
        if not self.compile_model:
            return module
        if not hasattr(torch, "compile"):
            if not self._compile_warned:
                print("Warning: torch.compile not available; skipping compilation.")
                self._compile_warned = True
            return module
        if self.device == "cpu" and not self.compile_on_cpu:
            if not self._compile_warned:
                print("Warning: torch.compile disabled on CPU; skipping compilation.")
                self._compile_warned = True
            return module
        compile_kwargs = {
            "backend": self.compile_backend,
            "mode": self.compile_mode,
        }
        try:
            return torch.compile(module, **compile_kwargs, dynamic=self.compile_dynamic)
        except TypeError:
            try:
                return torch.compile(module, **compile_kwargs)
            except Exception as exc:
                print(f"Warning: torch.compile failed for {name}: {exc}")
                return module
        except Exception as exc:
            print(f"Warning: torch.compile failed for {name}: {exc}")
            return module

    def _maybe_compile_modules(self):
        if not self.compile_model:
            return
        compile_targets = [
            "image_extractor",
            "resource_extractor",
            "entity_extractor",
            "fusion_attn",
            "lstm",
            "mlp",
            "critic",
        ]
        for name in compile_targets:
            module = getattr(self, name)
            compiled = self._compile_module(module, name)
            setattr(self, name, compiled)
        for name in compile_targets:
            getattr(self, name).train(self.train_mode)
        if not self.use_critic:
            self.critic.eval()

    def _fuse_features(self, image_features, resource_features, entity_features):
        fused = self.fusion_attn(image_features, resource_features, entity_features)
        return self.fusion_norm(fused)

    def _get_unwrapped(self, module):
        if hasattr(module, '_orig_mod'):
            return module._orig_mod
        return module

    def _normalize_point(self, point: Point2 | None) -> tuple[float, float]:
        if point is None:
            return 0.0, 0.0
        map_size = self.game_info.map_size
        x = float(point.x) / float(map_size.width)
        y = float(point.y) / float(map_size.height)
        return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))

    def _phase1_attack_target(self) -> Point2 | None:
        army = self.units.of_type(set(self.army_units)).ready
        if not army or len(army) < self._min_attack_units:
            return None

        if self.enemy_units:
            targets = [u.position for u in self.enemy_units]
            return army.center.closest(Point2(targets)) if targets else None

        if self.enemy_structures:
            townhall_types = {
                UnitTypeId.HATCHERY, UnitTypeId.LAIR, UnitTypeId.HIVE,
                UnitTypeId.NEXUS, UnitTypeId.COMMANDCENTER,
                UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS,
            }
            key_structures = list(self.enemy_structures.of_type(townhall_types))
            other_structures = list(self.enemy_structures.filter(lambda u: u.type_id not in townhall_types))
            all_targets = [s.position for s in key_structures + other_structures]
            return army.center.closest(Point2(all_targets)) if all_targets else None

        if self.enemy_start_locations:
            return army.center.closest(self.enemy_start_locations)

        return None

    def _default_attack_target(self) -> Point2 | None:
        army = self.units.of_type(set(self.army_units)).ready
        if not army:
            return None

        if self.enemy_units:
            return self.enemy_units.closest_to(army.center).position
        if self.enemy_structures:
            townhall_types = {
                UnitTypeId.HATCHERY, UnitTypeId.LAIR, UnitTypeId.HIVE,
                UnitTypeId.NEXUS, UnitTypeId.COMMANDCENTER,
                UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS,
            }
            key_targets = self.enemy_structures.of_type(townhall_types)
            target = (key_targets.first if key_targets else self.enemy_structures.first).position
            return target
        if self.enemy_start_locations:
            return self.enemy_start_locations[0]
        return None

    def _compute_param_targets(self, action_idx: int) -> torch.Tensor:
        targets = np.zeros(N_ACTION_PARAMS, dtype=np.float32)

        if action_idx in ACTION_PARAM_RANGES:
            lo, hi = ACTION_PARAM_RANGES[action_idx]
            if hi > lo:
                mid = (lo + hi) / 2.0
                targets[0] = float((mid - lo) / (hi - lo))
            else:
                targets[0] = 0.5
        else:
            targets[0] = 0.5

        if action_idx == 34 and self._phase == 1:
            targets[1] = 0.0
            target_point = self._phase1_attack_target()
        else:
            targets[1] = 0.5
            target_point = self._default_attack_target() if action_idx == 34 else None

        if action_idx == 34 and target_point is not None:
            targets[2], targets[3] = self._normalize_point(target_point)

        if action_idx == 40:
            retreat_point = self.townhalls.center if self.townhalls else self.start_location
            targets[2], targets[3] = self._normalize_point(retreat_point)

        return torch.as_tensor(targets, dtype=torch.float32)

    def load_checkpoint(self, checkpoint_path, load_optimizer: bool = True, load_scheduler: bool = True):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._get_unwrapped(self.image_extractor).load_state_dict(checkpoint['image_extractor'])
        self._get_unwrapped(self.resource_extractor).load_state_dict(checkpoint['resource_extractor'])
        self._get_unwrapped(self.entity_extractor).load_state_dict(checkpoint['entity_extractor'])
        if 'fusion_attn' in checkpoint:
            self._get_unwrapped(self.fusion_attn).load_state_dict(checkpoint['fusion_attn'])
        else:
            print(f"Warning: checkpoint missing fusion_attn: {checkpoint_path}")
        self.fusion_norm.load_state_dict(checkpoint['fusion_norm'])
        self._get_unwrapped(self.lstm).load_state_dict(checkpoint['lstm'])
        self._get_unwrapped(self.mlp).load_state_dict(checkpoint['mlp'])
        try:
            self.action_head.load_state_dict(checkpoint['action_head'])
        except RuntimeError as exc:
            print(
                f"Warning: action_head mismatch in {checkpoint_path}: {exc}. "
                "Keeping current action_head weights."
            )
        if 'param_head' in checkpoint:
            try:
                self.param_head.load_state_dict(checkpoint['param_head'])
            except RuntimeError as exc:
                print(
                    f"Warning: param_head mismatch in {checkpoint_path}: {exc}. "
                    "Keeping current param_head weights (fresh start for parameters)."
                )
        else:
            print(f"Note: checkpoint {checkpoint_path} has no param_head — using fresh weights.")
        self._get_unwrapped(self.critic).load_state_dict(checkpoint['critic'])
        if load_optimizer and 'optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except (ValueError, RuntimeError) as exc:
                print(
                    f"Warning: optimizer state mismatch in {checkpoint_path}: {exc}. "
                    "Skipping optimizer/scheduler load."
                )
                load_scheduler = False
        if load_scheduler and 'scheduler' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            except (ValueError, RuntimeError) as exc:
                print(
                    f"Warning: scheduler state mismatch in {checkpoint_path}: {exc}. "
                    "Skipping scheduler load."
                )
        if 'phase' in checkpoint:
            self._phase = checkpoint['phase']
        if 'total_actions' in checkpoint:
            self.total_actions = checkpoint['total_actions']
    def save_checkpoint(self, checkpoint_path):
        checkpoint = {
            'image_extractor': self._get_unwrapped(self.image_extractor).state_dict(),
            'resource_extractor': self._get_unwrapped(self.resource_extractor).state_dict(),
            'entity_extractor': self._get_unwrapped(self.entity_extractor).state_dict(),
            'fusion_attn': self._get_unwrapped(self.fusion_attn).state_dict(),
            'fusion_norm': self.fusion_norm.state_dict(),
            'lstm': self._get_unwrapped(self.lstm).state_dict(),
            'mlp': self._get_unwrapped(self.mlp).state_dict(),
            'action_head': self.action_head.state_dict(),
            'param_head': self.param_head.state_dict(),
            'critic': self._get_unwrapped(self.critic).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'phase': self._phase,
            'total_actions': self.total_actions,
        }
        torch.save(checkpoint, checkpoint_path)

    def expand_action_head(self, new_total_actions: int) -> None:
        """Expand action head from current size to new_total_actions.

        Preserves weights for indices 0-7 (core actions), initializes
        new indices with small random noise.
        """
        old_total = self.total_actions
        if new_total_actions <= old_total:
            print(f"Warning: new_total_actions ({new_total_actions}) <= old ({old_total}). No expansion.")
            return

        old_head_weight = self.action_head.linear.weight.data
        old_head_bias = self.action_head.linear.bias.data

        new_head_weight = torch.zeros(new_total_actions, old_head_weight.shape[1], device=old_head_weight.device)
        new_head_bias = torch.zeros(new_total_actions, device=old_head_bias.device)

        # Preserve core action indices 0-7
        preserve_indices = min(8, old_total)
        new_head_weight[:preserve_indices] = old_head_weight[:preserve_indices]
        new_head_bias[:preserve_indices] = old_head_bias[:preserve_indices]

        # Initialize new indices with small random noise
        with torch.no_grad():
            new_head_weight[preserve_indices:] = torch.randn(
                new_total_actions - preserve_indices,
                old_head_weight.shape[1],
                device=old_head_weight.device
            ) * 0.01
            new_head_bias[preserve_indices:] = torch.randn(
                new_total_actions - preserve_indices,
                device=old_head_bias.device
            ) * 0.01

        self.action_head.linear = nn.Linear(old_head_weight.shape[1], new_total_actions).to(self.device)
        with torch.no_grad():
            self.action_head.linear.weight.copy_(new_head_weight)
            self.action_head.linear.bias.copy_(new_head_bias)

        self.total_actions = new_total_actions
        print(f"Expanded action head from {old_total} to {new_total_actions} actions")
        print(f"  - Preserved weights for indices 0-{preserve_indices-1}")
        print(f"  - Initialized new indices {preserve_indices}-{new_total_actions-1} with random noise")

    @profile
    def choose_action(self, image, resources, entities, action_mask=None):
        image = torch.from_numpy(image).unsqueeze(0).to(self.device, dtype=torch.float32)
        resources = torch.from_numpy(resources).unsqueeze(0).to(self.device, dtype=torch.float32)
        entities = torch.from_numpy(entities).unsqueeze(0).to(self.device, dtype=torch.float32)
        image = torch.nan_to_num(image)
        resources = torch.nan_to_num(resources)
        entities = torch.nan_to_num(entities)
        if action_mask is None:
            action_mask = torch.ones(self.total_actions, dtype=torch.bool, device=self.device)
        else:
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        if not action_mask.any().item():
            action_mask = torch.zeros_like(action_mask)
            action_mask[0] = True
        if not torch.isfinite(self.lstm_state[0]).all() or not torch.isfinite(self.lstm_state[1]).all():
            self._reset_lstm_state()

        with torch.no_grad():
            image_features = self.image_extractor(image)
            resource_features = self.resource_extractor(resources)
            mask = (entities.abs().sum(dim=-1) > 1e-6)
            entity_features = self.entity_extractor(entities, mask=mask)

            combined = self._fuse_features(image_features, resource_features, entity_features)
            combined = combined.unsqueeze(1)
            lstm_out, self.lstm_state = self.lstm(combined, self.lstm_state)
            lstm_out = lstm_out.squeeze(1)

            self.lstm_state = (
                self.lstm_state[0].detach(),
                self.lstm_state[1].detach()
            )

            mlp_out = self.mlp(lstm_out)

            # ── Discrete action ──────────────────────────────────────────────
            logits = self.action_head.linear(mlp_out)
            logits_min = torch.finfo(logits.dtype).min
            logits = torch.nan_to_num(logits, nan=logits_min, posinf=logits_min, neginf=logits_min)
            logits = logits.masked_fill(~action_mask, logits_min)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_probs = dist.log_prob(action)

            # ── Continuous action parameters ─────────────────────────────────
            action_params, param_log_prob = self.param_head.sample(mlp_out)

            if self.use_critic:
                value = self.critic(lstm_out)
                return action, log_probs, value.squeeze(-1), action_params, param_log_prob
            value = torch.zeros((1,), device=self.device, dtype=torch.float32)
            return action, log_probs, value, action_params, param_log_prob

    def _estimate_bootstrap_value(self, image, resources, entities, mask, lstm_state=None):
        image_t = torch.as_tensor(image, dtype=torch.float32, device=self.device).unsqueeze(0)
        resources_t = torch.as_tensor(resources, dtype=torch.float32, device=self.device).unsqueeze(0)
        entities_t = torch.as_tensor(entities, dtype=torch.float32, device=self.device).unsqueeze(0)
        image_t = torch.nan_to_num(image_t)
        resources_t = torch.nan_to_num(resources_t)
        entities_t = torch.nan_to_num(entities_t)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        if lstm_state is None:
            lstm_state = self.lstm_state
        h0, c0 = lstm_state
        h0 = h0.detach()
        c0 = c0.detach()

        if not self.use_critic:
            return torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            image_features = self.image_extractor(image_t)
            resource_features = self.resource_extractor(resources_t)
            entity_features = self.entity_extractor(entities_t, mask=mask_t)
            combined = self._fuse_features(image_features, resource_features, entity_features)
            combined = combined.unsqueeze(1)
            lstm_out, _ = self.lstm(combined, (h0, c0))
            lstm_out = lstm_out.squeeze(1)
            value = self.critic(lstm_out)

        return value.squeeze(-1)

    def evaluate(self, image, resources, entities, mask, action, lstm_state):
        image_features = self.image_extractor(image)
        resource_features = self.resource_extractor(resources)
        entity_features = self.entity_extractor(entities, mask=mask)

        combined = self._fuse_features(image_features, resource_features, entity_features)

        combined = combined.unsqueeze(1)

        lstm_out, lstm_state = self.lstm(combined, lstm_state)
        lstm_out = lstm_out.squeeze(1)

        mlp_out = self.mlp(lstm_out)

        logits = self.action_head.linear(mlp_out)
        dist = torch.distributions.Categorical(logits=logits)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        if self.use_critic:
            value = self.critic(lstm_out)
            return log_prob, entropy, value.squeeze(-1), lstm_state
        value = torch.zeros((lstm_out.shape[0],), device=lstm_out.device, dtype=lstm_out.dtype)
        return log_prob, entropy, value, lstm_state
    
    @profile
    def ppo_update(
        self,
        epochs=5,
        eps_clip=None,
        bootstrap_value=None,
        minibatch_size=64,
        precompute_features=False,
    ):
        if eps_clip is None:
            eps_clip = self.clip_range
        if len(self.rollout_buffer.rewards) == 0:
            return
        mb_count = 0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        total_loss_sum = 0.0
        approx_kl_sum = 0.0
        clip_frac_sum = 0.0
        param_sup_loss_sum = 0.0
        explained_var_sum = 0.0
        explained_var_count = 0
        num_minibatches = max(1, len(self.rollout_buffer.rewards) // minibatch_size)
        images_buf    = torch.stack(self.rollout_buffer.images)
        resources_buf = torch.stack(self.rollout_buffer.resources)
        entities_buf  = torch.stack(self.rollout_buffer.entities)
        masks_buf     = torch.stack(self.rollout_buffer.masks)
        if len(self.rollout_buffer.action_masks) == len(self.rollout_buffer.rewards):
            action_masks_buf = torch.stack(self.rollout_buffer.action_masks)
        else:
            action_masks_buf = torch.ones(
                (len(self.rollout_buffer.rewards), self.total_actions),
                dtype=torch.bool,
                device=self.rollout_buffer.device,
            )

        target_device = torch.device(self.device)

        def _maybe_to_device(tensor, dtype=None):
            if tensor.device == target_device:
                return tensor if dtype is None else tensor.to(dtype=dtype)
            return tensor.to(device=target_device, dtype=dtype)

        actions       = _maybe_to_device(torch.stack(self.rollout_buffer.actions))
        old_log_probs = _maybe_to_device(torch.stack(self.rollout_buffer.log_probs))
        rewards       = _maybe_to_device(torch.stack(self.rollout_buffer.rewards))
        dones         = _maybe_to_device(torch.stack(self.rollout_buffer.dones))
        values        = _maybe_to_device(torch.stack(self.rollout_buffer.values)).squeeze(-1)

        # ── Action parameters (optional – buffer may not have them yet) ──────
        _buf_params    = getattr(self.rollout_buffer, 'action_params', None)
        _buf_plp       = getattr(self.rollout_buffer, 'param_log_probs', None)
        _buf_targets   = getattr(self.rollout_buffer, 'param_targets', None)
        has_params = (
            _buf_params is not None
            and len(_buf_params) == len(self.rollout_buffer.rewards)
        )
        has_param_targets = (
            _buf_targets is not None
            and len(_buf_targets) == len(self.rollout_buffer.rewards)
        )
        if has_params:
            old_action_params   = _maybe_to_device(torch.stack(_buf_params), dtype=torch.float32)
            old_param_log_probs = _maybe_to_device(torch.stack(_buf_plp),   dtype=torch.float32)
        else:
            old_action_params   = None
            old_param_log_probs = None
        if has_param_targets:
            param_targets = _maybe_to_device(torch.stack(_buf_targets), dtype=torch.float32)
        else:
            param_targets = None

        advantages, returns = self.gae(rewards, values, dones, bootstrap_value=bootstrap_value, gamma=self.gamma, lam=self.lam)
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns_norm = returns
        if self.normalize_returns:
            self.return_rms.update(returns)
            if self.return_rms.count >= 10:
                returns_norm = self.return_rms.normalize(returns)

        rollout_explained_var = None
        if self.use_critic:
            with torch.no_grad():
                ret_var_all = returns.var(unbiased=False)
                if ret_var_all > 1e-6:
                    rollout_explained_var = 1.0 - (returns - values).var(unbiased=False) / (ret_var_all + 1e-8)
                else:
                    rollout_explained_var = returns.new_tensor(0.0)

        T = len(rewards)
        seq_len = self.seq_len
        n_sequences = minibatch_size // seq_len
        start_indices = self._sample_valid_starts(T, seq_len, dones, n_sequences)

        if precompute_features:
            with torch.no_grad():
                all_img_feats = self.image_extractor(_maybe_to_device(images_buf, dtype=torch.float32))
                all_res_feats = self.resource_extractor(_maybe_to_device(resources_buf, dtype=torch.float32))
                all_ent_feats = self.entity_extractor(
                    _maybe_to_device(entities_buf, dtype=torch.float32),
                    _maybe_to_device(masks_buf, dtype=torch.float32),
                )
                all_combined = self._fuse_features(all_img_feats, all_res_feats, all_ent_feats)

        break_out = False
        for _ in range(epochs):
            # Do NOT shuffle start_indices — the LSTM relies on sequential h0/c0 states
            # saved during rollout collection. Shuffling breaks temporal context and
            # causes gradient chaos. Process slices in chronological order every epoch.
            epoch_starts = start_indices

            for start in epoch_starts:
                end = min(start + seq_len, T)
                sl = slice(start, end)

                h0 = self.rollout_buffer.h_states[start]
                c0 = self.rollout_buffer.c_states[start]
                h0 = _maybe_to_device(h0.view(self.n_lstm_layers, 1, self.lstm_hidden_size).detach())
                c0 = _maybe_to_device(c0.view(self.n_lstm_layers, 1, self.lstm_hidden_size).detach())

                if precompute_features:
                    combined = all_combined[sl]
                else:
                    img_feats = self.image_extractor(
                        _maybe_to_device(images_buf[sl], dtype=torch.float32)
                    )
                    res_feats = self.resource_extractor(
                        _maybe_to_device(resources_buf[sl], dtype=torch.float32)
                    )
                    ent_feats = self.entity_extractor(
                        _maybe_to_device(entities_buf[sl], dtype=torch.float32),
                        _maybe_to_device(masks_buf[sl], dtype=torch.float32),
                    )
                    combined = self._fuse_features(img_feats, res_feats, ent_feats)
                combined = combined.unsqueeze(0)

                lstm_out, _ = self.lstm(combined, (h0, c0))
                lstm_out = lstm_out.squeeze(0)

                mb_act    = actions[sl]
                mb_oldlp  = old_log_probs[sl]
                mb_adv    = advantages[sl]
                mb_ret    = returns_norm[sl]

                with autocast(device_type="cuda" if self.use_amp else "cpu", enabled=self.use_amp):
                    mlp_out = self.mlp(lstm_out)
                    logits  = self.action_head.linear(mlp_out)
                    mb_action_mask = _maybe_to_device(action_masks_buf[sl], dtype=torch.bool)
                    logits = logits.masked_fill(~mb_action_mask, torch.finfo(logits.dtype).min)  # match choose_action fill
                    dist    = torch.distributions.Categorical(logits=logits)

                    log_probs_new = dist.log_prob(mb_act)
                    entropies     = dist.entropy()
                    if self.use_critic:
                        new_values = self.critic(lstm_out).squeeze(-1)
                    else:
                        new_values = torch.zeros_like(mb_ret)

                    ratios = torch.exp(log_probs_new - mb_oldlp.detach())
                    surr1  = ratios * mb_adv
                    surr2  = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * mb_adv

                    policy_loss  = -torch.min(surr1, surr2).mean()
                    if self.use_critic:
                        old_values = values[sl]
                        if self._use_return_normalization and self.return_rms.count >= 10:
                            ret_mean = self.return_rms.mean.item()
                            ret_std = float(torch.sqrt(self.return_rms.var + 1e-8))
                            old_std = (old_values - ret_mean) / (ret_std + 1e-8)
                            new_std = (new_values - ret_mean) / (ret_std + 1e-8)
                            if self.clip_range_vf is not None:
                                v_std_clipped = old_std + torch.clamp(
                                    new_std - old_std, -self.clip_range_vf, self.clip_range_vf
                                )
                                value_loss_unclipped = (mb_ret - new_std).pow(2)
                                value_loss_clipped   = (mb_ret - v_std_clipped).pow(2)
                            else:
                                value_loss_unclipped = (mb_ret - new_std).pow(2)
                                value_loss_clipped   = value_loss_unclipped
                            new_values_for_ev = new_std
                        else:
                            value_loss_unclipped = (mb_ret - new_values).pow(2)
                            value_loss_clipped   = value_loss_unclipped
                            new_values_for_ev = new_values
                        value_loss   = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    else:
                        value_loss = policy_loss.new_tensor(0.0)
                    entropy_bonus = entropies.mean()
                    ent_coef = self._ent_coef_adaptive
                    if self.entropy_target is not None:
                        entropy_error = self.entropy_target - entropy_bonus.item()
                        self._ent_coef_adaptive = self._ent_coef_adaptive * (1.0 + 0.01 * entropy_error)
                        self._ent_coef_adaptive = max(1e-8, self._ent_coef_adaptive)
                        ent_coef = self._ent_coef_adaptive

                    # ── Parameter head loss (PPO-style for continuous params) ─
                    # We treat the parameter head as a Gaussian policy and apply
                    # the advantage-weighted log-prob loss: −A * log π(params|s).
                    # This teaches the model which param values lead to better outcomes.
                    param_mean, param_std = self.param_head(mlp_out)  # (T, n_params)
                    if has_params and old_action_params is not None:
                        mb_old_params   = old_action_params[sl]      # (T, n_params)
                        mb_old_param_lp = old_param_log_probs[sl]    # (T,)
                        param_dist     = torch.distributions.Normal(param_mean, param_std)
                        # Use raw logit of the stored sigmoid output as the sample
                        mb_params_raw  = torch.log(mb_old_params.clamp(1e-6, 1 - 1e-6) /
                                                   (1 - mb_old_params.clamp(1e-6, 1 - 1e-6)))
                        new_param_lp   = (param_dist.log_prob(mb_params_raw)
                                          - torch.log(mb_old_params * (1 - mb_old_params) + 1e-6)
                                         ).sum(dim=-1)
                        param_ratios   = torch.exp(new_param_lp - mb_old_param_lp.detach())
                        param_surr1    = param_ratios * mb_adv
                        param_surr2    = torch.clamp(param_ratios, 1 - eps_clip, 1 + eps_clip) * mb_adv
                        param_loss     = -torch.min(param_surr1, param_surr2).mean()
                    else:
                        param_loss = policy_loss.new_tensor(0.0)

                    if has_param_targets and param_targets is not None:
                        mb_param_targets = param_targets[sl]
                        param_supervised_loss = F.smooth_l1_loss(param_mean, mb_param_targets)
                    else:
                        param_supervised_loss = policy_loss.new_tensor(0.0)

                    loss = (
                        policy_loss
                        + self.vf_coef * value_loss
                        - ent_coef * entropy_bonus
                        + 0.5 * param_loss
                        + self.param_supervised_coef * param_supervised_loss
                    )

                with torch.no_grad():
                    approx_kl = (mb_oldlp - log_probs_new).mean()
                    if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                        break_out = True
                        break
                    clip_frac = (ratios - 1.0).abs().gt(eps_clip).float().mean()
                    if self.use_critic:
                        ret_var = mb_ret.var(unbiased=False)
                        if ret_var > 1e-6:
                            explained_var = 1.0 - (mb_ret - new_values_for_ev).var(unbiased=False) / (ret_var + 1e-8)
                            explained_var_sum += float(explained_var.detach().cpu())
                            explained_var_count += 1
                        else:
                            explained_var = mb_ret.new_tensor(0.0)
                    else:
                        explained_var = policy_loss.new_tensor(0.0)

                self.optimizer.zero_grad()
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.max_grad_norm)
                    self.optimizer.step()

                mb_count += 1
                policy_loss_sum += float(policy_loss.detach().cpu())
                value_loss_sum += float(value_loss.detach().cpu())
                entropy_sum += float(entropy_bonus.detach().cpu())
                total_loss_sum += float(loss.detach().cpu())
                approx_kl_sum += float(approx_kl.detach().cpu())
                clip_frac_sum += float(clip_frac.detach().cpu())
                param_sup_loss_sum += float(param_supervised_loss.detach().cpu())

            if break_out:
                break

        self._ppo_updates_total += 1
        self._ppo_updates_in_episode += 1
        if not self._warmup_completed and self.lr_warmup_steps > 0:
            warmup_factor = (self._ppo_updates_total + 1) / self.lr_warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self._base_lr * min(warmup_factor, 1.0)
            if self._ppo_updates_total >= self.lr_warmup_steps:
                self._warmup_completed = True
        elif self.scheduler is not None:
            self.scheduler.step()

        if self.log_mlflow:
            denom = max(1, mb_count)
            ev_denom = max(1, explained_var_count)
            mlflow.log_metrics(
                {
                    "ppo/policy_loss": policy_loss_sum / denom,
                    "ppo/value_loss": value_loss_sum / denom,
                    "ppo/entropy": entropy_sum / denom,
                    "ppo/total_loss": total_loss_sum / denom,
                    "ppo/approx_kl": approx_kl_sum / denom,
                    "ppo/clip_frac": clip_frac_sum / denom,
                    "ppo/param_supervised_loss": param_sup_loss_sum / denom,
                    "ppo/explained_var": explained_var_sum / ev_denom,
                    "ppo/explained_var_rollout": float(rollout_explained_var.detach().cpu()) if rollout_explained_var is not None else 0.0,
                    "ppo/adv_mean": float(advantages.mean().detach().cpu()),
                    "ppo/adv_std": float(advantages.std().detach().cpu()),
                    "ppo/updates_total": float(self._ppo_updates_total),
                    "ppo/updates_in_episode": float(self._ppo_updates_in_episode),
                    "ppo/learning_rate": self.scheduler.get_last_lr()[0] if self.scheduler is not None else self._base_lr,
                },
                step=self._mlflow_step,
            )
            self._mlflow_step += 1

        self.rollout_buffer.reset()
        # Also clear the action-parameter extension lists
        self.rollout_buffer.action_params   = []
        self.rollout_buffer.param_log_probs = []
        self.rollout_buffer.param_targets   = []

    async def on_start(self):
        self.client.game_step = 4
        self._prev_metrics = self._collect_metrics()
        self._prev_minerals = self.minerals
        self._prev_time = self.time
        self._episode_reward_sum = 0.0
        self._episode_reward_count = 0
        self._reset_lstm_state()
        self._milestone_16_achieved = False
        self._milestone_32_achieved = False
        self._milestone_48_achieved = False
        self._milestone_60_achieved = False
        self.reset_cumulative_stats()
        self._pending_update = False
        self.step = 0
        self._ppo_updates_in_episode = 0
        # Ensure rollout buffer has lists for action parameters
        if not hasattr(self.rollout_buffer, 'action_params'):
            self.rollout_buffer.action_params   = []
            self.rollout_buffer.param_log_probs = []
        if not hasattr(self.rollout_buffer, 'param_targets'):
            self.rollout_buffer.param_targets = []
        
    @profile
    async def on_step(self, iteration):
        image, resources, entities = self.get_observation()
        entities_np = np.asarray(entities, dtype=np.float32)
        if entities_np.ndim == 1:
            if entities_np.size == 0:
                entities_np = entities_np.reshape(0, 11)
            else:
                entities_np = entities_np.reshape(1, -1)
        max_units = self.max_units_tracked
        if entities_np.shape[0] < max_units:
            pad = np.zeros((max_units - entities_np.shape[0], entities_np.shape[1]), dtype=np.float32)
            entities_np = np.concatenate([entities_np, pad], axis=0)
        elif entities_np.shape[0] > max_units:
            entities_np = entities_np[:max_units]
        mask = (np.abs(entities_np).sum(axis=-1) > 1e-6).astype(np.float32)

        if self.train_mode and self._pending_update:
            bootstrap_value = self._estimate_bootstrap_value(
                image,
                resources,
                entities_np,
                mask,
                lstm_state=self.lstm_state,
            )
            self.ppo_update(bootstrap_value=bootstrap_value, minibatch_size=self.minibatch_size, epochs=self.ppo_update_epochs)
            self._pending_update = False

        prev_lstm_state = self.lstm_state
        action_mask = await self.get_action_mask()
        action, log_prob, value, action_params, param_log_prob = self.choose_action(
            image,
            resources,
            entities_np,
            action_mask=action_mask,
        )

        # FIX: execute action and compute reward BEFORE any buffer/update logic
        success = await self._execute_action(action.item(), action_params=action_params.squeeze(0).cpu().numpy())
        reward  = self._compute_step_reward(success, action.item())

        self._log_step(iteration, action.item(), success, reward)

        self._episode_reward_sum += float(reward)
        self._episode_reward_count += 1

        self.step = iteration
        done = 1 if self.client._game_result is not None else 0
        if done:
            self._reset_lstm_state()

        if self.train_mode:
            self.rollout_buffer.add(
                obs=(image, resources, entities_np, mask),
                action=action.item(),
                log_prob=log_prob.detach(),
                reward=reward,
                done=done,
                value=value.detach(),
                lstm_state=prev_lstm_state,
                action_mask=action_mask,
            )
            # Append action parameters (not part of base RolloutBuffer API)
            if not hasattr(self.rollout_buffer, 'action_params'):
                self.rollout_buffer.action_params   = []
                self.rollout_buffer.param_log_probs = []
            if not hasattr(self.rollout_buffer, 'param_targets'):
                self.rollout_buffer.param_targets = []
            self.rollout_buffer.action_params.append(action_params.detach())
            self.rollout_buffer.param_log_probs.append(param_log_prob.detach())
            self.rollout_buffer.param_targets.append(self._compute_param_targets(action.item()).to(self.rollout_buffer.device))

            if len(self.rollout_buffer.rewards) >= self.rollout_size:
                self._pending_update = True
            
    async def on_end(self, result):
        # FIX: terminal rewards scaled from ±50/−25 to ±5/−2.5 to match dense reward
        # magnitudes (~0.05–1.0 per step). The old scale caused the value function to
        # ignore all shaped rewards and only predict win/loss, collapsing explained_var.
        if result == Result.Victory:
            final_reward = 50.0
        elif result == Result.Defeat:
            final_reward = -50.0
        else:
            final_reward = -10.0

        if self.train_mode and len(self.rollout_buffer.rewards) > 0:
            self.rollout_buffer.rewards[-1] += final_reward
            self.rollout_buffer.dones[-1] = torch.as_tensor(
                1.0,
                dtype=torch.float32,
                device=self.rollout_buffer.device,
            )
        if self.log_mlflow:
            episode_reward_sum = self._episode_reward_sum + float(final_reward)
            episode_reward_count = max(1, self._episode_reward_count)
            mlflow.log_metrics(
                {
                    "env/final_reward": float(final_reward),
                    "env/episode_avg_reward": float(episode_reward_sum / episode_reward_count),
                    "env/result": float(1 if result == Result.Victory else 0),
                    "env/ppo_updates_in_episode": float(self._ppo_updates_in_episode),
                    "env/episode_length": float(self.step + 1),
                },
                step=self._mlflow_step,
            )
        self._episode_reward_sum = 0.0
        self._episode_reward_count = 0
        self._pending_update = False
        if self.train_mode and len(self.rollout_buffer.rewards) > 0:
            self.ppo_update(bootstrap_value=torch.tensor(0.0, device=self.device), minibatch_size=self.minibatch_size, epochs=self.ppo_update_epochs)