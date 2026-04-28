from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable

import sc2
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
import os
import sys
import importlib.util
import inspect
from typing import Optional

from scaffold import Scaffold
from staragent import StarAgent


@dataclass
class RewardStats:
    values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.values.append(float(value))

    def summary(self) -> str:
        if not self.values:
            return "no rewards recorded"

        min_reward = min(self.values)
        max_reward = max(self.values)
        avg_reward = mean(self.values)
        nonzero = sum(1 for value in self.values if value != 0.0)

        return (
            f"count={len(self.values)} min={min_reward:.4f} max={max_reward:.4f} "
            f"mean={avg_reward:.4f} nonzero={nonzero}"
        )


class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        self._file = None
        self._writer = None
        self._open()

    def _open(self) -> None:
        is_new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        self._file = open(self.path, "a", newline="")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(
                [
                    "match",
                    "iteration",
                    "reward",
                    "action",
                    "action_succeeded",
                    "agent",
                    "opponent",
                ]
            )

    def log(self, match_idx: int, iteration: int, reward: float, action: str, action_succeeded: int, agent: str, opponent: str) -> None:
        if self._writer is None:
            return
        self._writer.writerow(
            [
                match_idx,
                iteration,
                f"{reward:.6f}",
                action,
                action_succeeded,
                agent,
                opponent,
            ]
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


@dataclass
class RewardSnapshot:
    workers: float = 0.0
    army: float = 0.0
    minerals: float = 0.0
    gas: float = 0.0
    enemy_unit_kills: float = 0.0
    enemy_structures_destroyed: float = 0.0
    lost_army: float = 0.0
    lost_workers: float = 0.0
    hatcheries: float = 0.0
    supply_used: float = 0.0
    supply_cap: float = 0.0
    game_time: float = 0.0
    structures: float = 0.0
    has_queen: bool = False
    supply_left: float = 0.0
    enemy_base_distance: float | None = None
    idle_army: bool = False


def compute_reward(
    current: RewardSnapshot,
    previous: RewardSnapshot,
    reward_weights: dict[str, float],
    *,
    action_succeeded: bool,
    action_idx: int,
    milestones: dict[str, bool],
) -> float:
    delta_workers = current.workers - previous.workers
    delta_army = current.army - previous.army
    delta_enemy_unit_kills = current.enemy_unit_kills - previous.enemy_unit_kills
    delta_enemy_structures_destroyed = current.enemy_structures_destroyed - previous.enemy_structures_destroyed
    delta_losses = current.lost_army - previous.lost_army
    delta_worker_loss = current.lost_workers - previous.lost_workers
    delta_hatcheries = current.hatcheries - previous.hatcheries
    delta_structures = current.structures - previous.structures

    reward = 0.0
    reward += reward_weights.get("workers", 0.0) * delta_workers
    reward += reward_weights.get("army", 0.0) * delta_army
    reward += reward_weights.get("losses", 0.0) * delta_losses
    reward += reward_weights.get("worker_loss", 0.0) * delta_worker_loss
    reward += reward_weights.get("enemy_unit_kills", 0.0) * delta_enemy_unit_kills
    reward += reward_weights.get("enemy_structures_destroyed", 0.0) * delta_enemy_structures_destroyed
    reward += reward_weights.get("expansion", 0.0) * delta_hatcheries

    if action_succeeded:
        reward += reward_weights.get("success", 0.0)

    if current.has_queen:
        reward += reward_weights.get("queen_exists", 0.0)

    ideal_workers = current.hatcheries * 16
    saturation_gap = abs(current.workers - ideal_workers)
    if ideal_workers > 0:
        saturation_penalty = saturation_gap / float(max(1.0, ideal_workers))
    else:
        saturation_penalty = saturation_gap
    reward -= reward_weights.get("worker_saturation", 0.0) * saturation_penalty

    if current.hatcheries > 0:
        workers_per_base = current.workers / current.hatcheries
        if workers_per_base >= 16:
            reward += reward_weights.get("workers_per_hatchery", 0.0)

    if action_succeeded and action_idx == 3:
        reward += reward_weights.get("attack_action", 0.0)

    if delta_structures > 0:
        reward += reward_weights.get("structure_built", 0.0)

    if current.workers < 5 and current.hatcheries >= 1:
        reward += reward_weights.get("low_worker_penalty", 0.0)

    if current.supply_left <= 0:
        reward += reward_weights.get("supply_penalty", 0.0)

    if current.enemy_base_distance is not None and current.enemy_base_distance < 20:
        reward += reward_weights.get("enemy_base_proximity", 0.0) * (1.0 - current.enemy_base_distance / 20.0)

    if current.idle_army and action_idx in (27, 28):
        reward += reward_weights.get("army_movement", 0.0)

    wn = current.workers
    # Reduced milestone set: only keep larger, meaningful milestones
    if wn >= 32 and not milestones.get("32", False):
        milestones["32"] = True
        reward += reward_weights.get("worker_milestone_32", 0.0)
    if wn >= 60 and not milestones.get("60", False):
        milestones["60"] = True
        reward += reward_weights.get("worker_milestone_60", 0.0)

    # No clipping in probe: return raw shaped reward
    return float(reward)


def offline_reward_report(reward_weights: dict[str, float]) -> None:
    print("[RewardProbe] offline reward scenarios")

    scenarios: list[tuple[str, RewardSnapshot, RewardSnapshot, bool, int]] = [
        (
            "baseline / no change",
            RewardSnapshot(workers=12, army=0, hatcheries=1, supply_left=5),
            RewardSnapshot(workers=12, army=0, hatcheries=1, supply_left=5),
            False,
            0,
        ),
        (
            "+1 worker",
            RewardSnapshot(workers=13, army=0, hatcheries=1, supply_left=4),
            RewardSnapshot(workers=12, army=0, hatcheries=1, supply_left=5),
            True,
            4,
        ),
        (
            "+1 army unit",
            RewardSnapshot(workers=12, army=1, hatcheries=1, supply_left=4),
            RewardSnapshot(workers=12, army=0, hatcheries=1, supply_left=5),
            True,
            2,
        ),
        (
            "+1 structure / expansion",
            RewardSnapshot(workers=12, army=0, hatcheries=2, structures=3, supply_left=5),
            RewardSnapshot(workers=12, army=0, hatcheries=1, structures=2, supply_left=5),
            True,
            36,
        ),
        (
            "queen exists",
            RewardSnapshot(workers=18, army=0, hatcheries=1, has_queen=True, supply_left=6),
            RewardSnapshot(workers=18, army=0, hatcheries=1, supply_left=6),
            False,
            0,
        ),
        (
            "attack success",
            RewardSnapshot(workers=20, army=6, hatcheries=2, idle_army=True, enemy_base_distance=10, supply_left=3),
            RewardSnapshot(workers=20, army=6, hatcheries=2, idle_army=True, enemy_base_distance=12, supply_left=3),
            True,
            3,
        ),
        (
            "supply blocked",
            RewardSnapshot(workers=18, army=4, hatcheries=1, supply_left=0),
            RewardSnapshot(workers=18, army=4, hatcheries=1, supply_left=1),
            False,
            0,
        ),
        (
            "worker milestones burst",
            RewardSnapshot(workers=60, army=0, hatcheries=4, supply_left=8),
            RewardSnapshot(workers=15, army=0, hatcheries=4, supply_left=8),
            True,
            4,
        ),
        (
            "enemy structure kill",
            RewardSnapshot(workers=24, army=8, hatcheries=2, enemy_structures_destroyed=1, enemy_base_distance=8, supply_left=4),
            RewardSnapshot(workers=24, army=8, hatcheries=2, enemy_structures_destroyed=0, enemy_base_distance=8, supply_left=4),
            True,
            3,
        ),
    ]

    print("[RewardProbe] reward clipping: none (raw shaped rewards are returned)")
    for name, current, previous, action_succeeded, action_idx in scenarios:
        milestones: dict[str, bool] = {}
        reward = compute_reward(
            current,
            previous,
            reward_weights,
            action_succeeded=action_succeeded,
            action_idx=action_idx,
            milestones=milestones,
        )
        print(f"[RewardProbe] {name:<24} reward={reward:+.4f}")

    print("[RewardProbe] weight summary")
    for key in sorted(reward_weights):
        print(f"[RewardProbe]   {key}: {reward_weights[key]:+.4f}")


def discover_bots(bots_dir: str = "bots") -> list[str]:
    candidates = []
    root = os.path.abspath(bots_dir)
    if not os.path.isdir(root):
        return candidates
    for sub in os.listdir(root):
        subpath = os.path.join(root, sub)
        if not os.path.isdir(subpath):
            continue
        for fname in os.listdir(subpath):
            if not fname.endswith(".py"):
                continue
            if fname == "__init__.py":
                continue
            rel = os.path.join("bots", sub, fname)
            candidates.append(rel)
    return sorted(candidates)


def load_bot_class_from_path(path: str) -> Optional[type]:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return None
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
    except Exception:
        return None
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        try:
            if issubclass(obj, sc2.bot_ai.BotAI) and obj is not sc2.bot_ai.BotAI:
                return obj
        except Exception:
            continue
    return None


def infer_race_from_path(path: str) -> Race:
    norm = path.replace("\\", "/")
    if "/zerg/" in norm:
        return Race.Zerg
    if "/protoss/" in norm:
        return Race.Protoss
    if "/terran/" in norm:
        return Race.Terran
    return Race.Random


def build_scripted_probe_class(bot_class: type) -> type:
    class ScriptedRewardProbe(bot_class, Scaffold):
        def __init__(self, csv_logger: CSVLogger | None = None, agent_name: str = "scripted", opponent_name: str = "opponent", match_idx: int = 1):
            Scaffold.__init__(self, log_level=0)
            try:
                bot_class.__init__(self)
            except Exception:
                pass
            self.reward_stats = RewardStats()
            self.csv_logger = csv_logger
            self.agent_name = agent_name
            self.opponent_name = opponent_name
            self.match_idx = match_idx

        async def on_start(self):
            if hasattr(bot_class, "on_start"):
                await bot_class.on_start(self)
            self.reset_cumulative_stats()

        async def on_step(self, iteration: int):
            if hasattr(bot_class, "on_step"):
                await bot_class.on_step(self, iteration)
            reward = self._compute_step_reward(False, -1)
            self.reward_stats.add(reward)
            if self.csv_logger is not None:
                self.csv_logger.log(
                    self.match_idx,
                    iteration,
                    reward,
                    "scripted",
                    0,
                    self.agent_name,
                    self.opponent_name,
                )
            print(
                f"[RewardProbe] step_reward={reward:.4f} action=scripted "
                f"iter={iteration}"
            )

        async def on_end(self, result):
            if hasattr(bot_class, "on_end"):
                await bot_class.on_end(self, result)
            print(f"[RewardProbe] game_result={result.name}")
            print(f"[RewardProbe] reward_summary: {self.reward_stats.summary()}")

    ScriptedRewardProbe.__name__ = f"ScriptedRewardProbe_{bot_class.__name__}"
    return ScriptedRewardProbe


class RewardProbeBot(StarAgent):
    def __init__(self, csv_logger: CSVLogger | None = None, agent_name: str = "staragent", opponent_name: str = "opponent", match_idx: int = 1):
        super().__init__(train_mode=False, log_mlflow=False)
        self.reward_stats = RewardStats()
        self.csv_logger = csv_logger
        self.agent_name = agent_name
        self.opponent_name = opponent_name
        self.match_idx = match_idx

    def _compute_step_reward(self, action_succeeded: bool, action_idx: int = -1) -> float:
        reward = super()._compute_step_reward(action_succeeded, action_idx)
        self.reward_stats.add(reward)
        if self.csv_logger is not None:
            self.csv_logger.log(
                self.match_idx,
                len(self.reward_stats.values),
                reward,
                str(action_idx),
                int(bool(action_succeeded)),
                self.agent_name,
                self.opponent_name,
            )
        print(
            f"[RewardProbe] step_reward={reward:.4f} action={action_idx} "
            f"success={int(bool(action_succeeded))}"
        )
        return reward

    async def on_end(self, result):
        await super().on_end(result)
        print(f"[RewardProbe] game_result={result.name}")
        print(f"[RewardProbe] reward_summary: {self.reward_stats.summary()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short match and print reward range.")
    parser.add_argument("--map", default="CeruleanFallLE", help="SC2 map name")
    parser.add_argument("--difficulty", default="Easy", help="Computer difficulty")
    parser.add_argument("--time-limit", type=int, default=5 * 60, help="Game time limit in seconds")
    parser.add_argument("--offline", action="store_true", help="Print synthetic reward scenarios instead of running SC2")
    parser.add_argument("--list", action="store_true", help="List available bots under bots/ and exit")
    parser.add_argument("--agent", default=None, help="Relative path to scripted bot for the probe agent (e.g. bots/zerg/zerg_rush.py)")
    parser.add_argument("--opponent", default=None, help="Relative path to opponent bot file under bots/ (e.g. bots/zerg/zerg_rush.py)")
    parser.add_argument("--matches", type=int, default=1, help="Number of matches to run vs the opponent")
    parser.add_argument("--csv", default="reward_probe.csv", help="CSV output path (default: reward_probe.csv)")
    parser.add_argument("--no-csv", action="store_true", help="Disable CSV output")
    args = parser.parse_args()

    if args.list:
        bots = discover_bots()
        print("[RewardProbe] available bots:")
        for b in bots:
            print(f"  - {b}")
        return

    if args.offline:
        offline_reward_report(Scaffold()._reward_weights)
        return

    difficulty = getattr(Difficulty, args.difficulty)
    bots_list = discover_bots()

    agent_path = args.agent
    if agent_path is None and bots_list:
        agent_path = bots_list[-1]

    agent_race = Race.Zerg
    agent_bot = None
    if agent_path is not None:
        agent_file = os.path.abspath(agent_path)
        agent_race = infer_race_from_path(agent_file)
        AgentClass = load_bot_class_from_path(agent_file)
        if AgentClass is None:
            print(f"[RewardProbe] failed to load agent bot from {agent_file}; using StarAgent instead")
        else:
            agent_bot = build_scripted_probe_class(AgentClass)()

    csv_logger = None if args.no_csv else CSVLogger(args.csv)

    if agent_bot is None:
        agent_bot = RewardProbeBot(csv_logger=csv_logger)

    opponent_path = args.opponent
    if opponent_path is None:
        opponent_path = None if not bots_list else bots_list[4]

    OppClass = None
    opp_race = Race.Random
    if opponent_path is not None:
        opponent_file = os.path.abspath(opponent_path)
        opp_race = infer_race_from_path(opponent_file)
        OppClass = load_bot_class_from_path(opponent_file)
        if OppClass is None:
            print(f"[RewardProbe] failed to load opponent bot from {opponent_file}; falling back to Computer")

    print(
        f"[RewardProbe] running map={args.map} difficulty={difficulty.name} "
        f"time_limit={args.time_limit}s agent={agent_path} opponent={opponent_path}"
    )
    for match_i in range(args.matches):
        try:
            if opponent_path is None or OppClass is None:
                opponent_player = Computer(Race.Random, difficulty)
            else:
                opponent_player = Bot(race=opp_race, ai=OppClass())

            # Recreate the agent bot each match to reset state
            agent_name = os.path.basename(agent_path) if agent_path else "staragent"
            opponent_name = os.path.basename(opponent_path) if opponent_path else f"Computer_{difficulty.name}"
            if agent_bot is not None and agent_path is not None and agent_bot.__class__.__name__.startswith("ScriptedRewardProbe"):
                agent_class = load_bot_class_from_path(os.path.abspath(agent_path))
                if agent_class is not None:
                    agent_bot = build_scripted_probe_class(agent_class)(
                        csv_logger=csv_logger,
                        agent_name=agent_name,
                        opponent_name=opponent_name,
                        match_idx=match_i + 1,
                    )
            else:
                agent_bot = RewardProbeBot(
                    csv_logger=csv_logger,
                    agent_name=agent_name,
                    opponent_name=opponent_name,
                    match_idx=match_i + 1,
                )

            result = run_game(
                sc2.maps.get(args.map),
                [Bot(agent_race, agent_bot), opponent_player],
                realtime=False,
                game_time_limit=args.time_limit,
            )
            print(f"[RewardProbe] match={match_i+1}/{args.matches} result={result}")
            print(f"[RewardProbe] final_summary: {agent_bot.reward_stats.summary()}")
        except Exception as exc:
            print(f"[RewardProbe] live SC2 match failed: {exc}")
            offline_reward_report(agent_bot._reward_weights)

    if csv_logger is not None:
        csv_logger.close()


if __name__ == "__main__":
    main()