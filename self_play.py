import json
import logging
import random
import signal
import os
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import mlflow
from utils import update_elo, evaluate_agent, plot_radar_chart
import sc2
from sc2.data import Difficulty, Race, Result
from sc2.main import run_game
from sc2.player import Bot, Computer

from bots.protoss.cannon_rush import CannonRushBot
from bots.protoss.threebase_voidray import ThreebaseVoidrayBot
from bots.protoss.warpgate_push import WarpGateBot
from bots.terran.cyclone_push import CyclonePush
from bots.terran.mass_reaper import MassReaperBot
from bots.terran.onebase_battlecruiser import BCRushBot
from bots.terran.proxy_rax import ProxyRaxBot
from bots.zerg.hydralisk_push import Hydralisk
from bots.zerg.onebase_broodlord import BroodlordBot
from bots.zerg.zerg_rush import ZergRushBot
from staragent import StarAgent


class PortpickerFilter(logging.Filter):
    def filter(self, record):
        return (
            "Returning a port that wasn't given by portpicker"
            not in record.getMessage()
        )


logging.getLogger().addFilter(PortpickerFilter())


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[35m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record):
        original_levelname = record.levelname
        color = self.COLORS.get(original_levelname, self.RESET)
        record.levelname = f"{color}{original_levelname}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


logger = logging.getLogger("self_play")
logger.setLevel(logging.INFO)
logger.handlers.clear()
handler = logging.StreamHandler()
formatter = ColoredFormatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


TIMEOUT = 30 * 60
SAVE_EPISODES = 10
ROUNDS = 1000
CHKPT_DIR = Path("checkpoints")
MAPS_DIR = Path("StarCraftII/maps")
ELO_RATINGS_PATH = Path("elo_ratings.json")
EXPERIMENT_NAME = "starcraft-selfplay"
RECENT_WINRATE_WINDOWS = [10, 20]
RECENT_SELFPLAY_WINDOWS = [10, 20]
SELFPLAY_SAVE_WINDOW = 10

os.makedirs("charts", exist_ok=True)

MEAT_WALLS = [
    Computer(Race.Random, Difficulty.VeryEasy),
    Computer(Race.Random, Difficulty.Easy),
    Computer(Race.Random, Difficulty.Medium),
    Computer(Race.Random, Difficulty.MediumHard),
    Computer(Race.Random, Difficulty.Hard),
    Computer(Race.Random, Difficulty.Harder),
    Computer(Race.Random, Difficulty.VeryHard),
    Computer(Race.Random, Difficulty.CheatVision),
    Computer(Race.Random, Difficulty.CheatMoney),
    Computer(Race.Random, Difficulty.CheatInsane),
]

MEAT_WALLS_NAME = [
    "VeryEasy",
    "Easy",
    "Medium",
    "MediumHard",
    "Hard",
    "Harder",
    "VeryHard",
    "CheatVision",
    "CheatMoney",
    "CheatInsane",
]

ELO_INITIAL = 1200

SCRIPTED_BOTS = [
    (Race.Zerg, Hydralisk),
    (Race.Zerg, BroodlordBot),
    (Race.Zerg, ZergRushBot),
    (Race.Protoss, CannonRushBot),
    (Race.Protoss, ThreebaseVoidrayBot),
    (Race.Protoss, WarpGateBot),
    (Race.Terran, CyclonePush),
    (Race.Terran, MassReaperBot),
    (Race.Terran, BCRushBot),
    (Race.Terran, ProxyRaxBot),
]

should_stop = False

def signal_handler(signum, frame):
    del signum, frame
    global should_stop
    logger.warning("Received Ctrl+C, finishing current round then stopping...")
    should_stop = True

def log_elo_artifact(elo_ratings):
    with ELO_RATINGS_PATH.open("w") as f:
        json.dump(elo_ratings, f, indent=2, sort_keys=True)
    mlflow.log_artifact(str(ELO_RATINGS_PATH))


def _init_recent_windows(windows):
    return {int(window): deque(maxlen=int(window)) for window in windows}


def _update_recent_windows(recent_map, result):
    for window in recent_map.values():
        window.append(result)


def _win_rate_from_results(results):
    if not results:
        return 0.0
    wins = sum(1 for r in results if r == 1)
    return wins / len(results)


def _log_recent_winrates(prefix, recent_map, round_num, label=None):
    if not recent_map:
        return
    for window in sorted(recent_map):
        rate = _win_rate_from_results(recent_map[window])
        if label is None:
            name = f"{prefix}/win_rate_last_{window}"
        else:
            name = f"{prefix}/{label}_win_rate_last_{window}"
        mlflow.log_metric(name, rate, step=round_num)


def _checkpoint_version(path: Path):
    stem = path.stem
    if "_v" not in stem:
        return None
    version_str = stem.split("_v", 1)[1]
    if not version_str.isdigit():
        return None
    return int(version_str)


def _checkpoint_prefix(path: Path) -> str:
    """Extract the phase prefix from checkpoint name, e.g. 'p1' from 'hannibal_p1_v12.pt'"""
    stem = path.stem
    for part in stem.split("_"):
        if part.startswith("p") and len(part) > 1 and part[1:].isdigit():
            return part
    return ""


def _find_checkpoints():
    checkpoints = []
    for path in CHKPT_DIR.glob("hannibal_p*.pt"):
        version = _checkpoint_version(path)
        if version is not None:
            checkpoints.append((version, path))
    if not checkpoints:
        # Fallback to old format for resuming old runs
        for path in CHKPT_DIR.glob("hannibal_v*.pt"):
            version = _checkpoint_version(path)
            if version is not None:
                checkpoints.append((version, path))
    return [path for _, path in sorted(checkpoints, key=lambda item: item[0])]


def _load_elo_ratings():
    if not ELO_RATINGS_PATH.exists():
        return {}
    with ELO_RATINGS_PATH.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data

def main():
    global should_stop

    should_stop = False
    CHKPT_DIR.mkdir(parents=True, exist_ok=True)
    maps = sorted(
        path.stem for path in MAPS_DIR.rglob("*.SC2Map")
        if "mini_games" not in path.parts
    )
    logger.info(f"Found {len(maps)} maps")
    if not maps:
        raise AssertionError(f"No SC2Map files found in {MAPS_DIR}")
    signal.signal(signal.SIGINT, signal_handler)

    meat_level = 0
    v = 1
    versions = []
    elo_ratings = _load_elo_ratings()
    recent_results = _init_recent_windows(RECENT_WINRATE_WINDOWS)
    recent_selfplay_results = _init_recent_windows(RECENT_SELFPLAY_WINDOWS)
    stats = {
        "wins": 0,
        "ties": 0,
        "losses": 0,
    }

    map_rng = random.Random()
    round_num = 0

    mlflow.set_experiment(EXPERIMENT_NAME)
    if mlflow.active_run() is not None:
        mlflow.end_run()

    mlflow.start_run()

    hannibal_agent = StarAgent(
        train_mode=True,
        name="Hannibal",
        log_mlflow=True,
        use_critic=True,
        normalize_returns=True,
        compile_model=True,
        phase=1,
    )
    hannibal = Bot(Race.Zerg, hannibal_agent, name="Hannibal")
    checkpoint_paths = _find_checkpoints()
    if checkpoint_paths:
        latest_checkpoint = checkpoint_paths[-1]
        latest_version = _checkpoint_version(latest_checkpoint)
        if latest_version is not None:
            v = latest_version
        hannibal_agent.load_checkpoint(latest_checkpoint, load_optimizer=True, load_scheduler=True)
        versions = [path.name for path in checkpoint_paths]
        versions.sort(key=lambda key: elo_ratings.get(key, ELO_INITIAL), reverse=True)
        logger.info(f"Resuming from {latest_checkpoint.name}")
    else:
        initial_checkpoint = CHKPT_DIR / f"hannibal_v{v}.pt"
        hannibal_agent.save_checkpoint(initial_checkpoint)
        versions.append(initial_checkpoint.name)
    elo_ratings.setdefault(hannibal.name, ELO_INITIAL)
    for name in versions:
        elo_ratings.setdefault(name, ELO_INITIAL)

    scipio_agent = StarAgent(
        train_mode=False,
        name="Scipio",
        log_mlflow=False,
        compile_model=True,
        phase=1,
    )
    scipio = Bot(Race.Zerg, scipio_agent, name="Scipio")

    for round_num in range(1, ROUNDS + 1):
        if should_stop:
            logger.info("Stopping after current round completes.")
            break

        # Set the current round number and phase for action masking
        hannibal_agent.set_round_number(round_num)
        current_phase = hannibal_agent._phase

        # Phase-based opponent weights
        if current_phase == 1:
            weights = [0.05, 0.45, 0.50]  # scipio, scripted, meatwall
        elif current_phase == 2:
            weights = [0.30, 0.35, 0.35]
        else:  # phase 3+
            weights = [0.60, 0.20, 0.20]

        # Filter versions by phase for scipio opponent pool
        phase_prefix = f"_p{current_phase}"
        phase_versions = [v for v in versions if phase_prefix in v]
        if not phase_versions:
            phase_versions = versions[:3]  # fallback to top 3

        opponent_type = random.choices(
            ["scipio", "scripted", "meatwall"],
            weights=weights,
        )[0]

        map = map_rng.choice(maps)

        if opponent_type == "scipio":
            scipio_v = random.choice(phase_versions)
            # Set scipio_agent phase to match checkpoint to avoid head size mismatch
            # Default to current_phase if not found in filename
            scipio_checkpoint_phase = current_phase
            for part in scipio_v.split("_"):
                if part.startswith("p") and len(part) > 1 and part[1:].isdigit():
                    scipio_checkpoint_phase = int(part[1:])
                    break
            
            # Set phase and resize action head BEFORE loading checkpoint
            scipio_agent._phase = scipio_checkpoint_phase
            if scipio_checkpoint_phase == 1:
                scipio_agent.total_actions = 8
            elif scipio_checkpoint_phase == 2:
                scipio_agent.total_actions = 15
            # Phase 3 uses full 41 actions by default
            
            scipio_agent.load_checkpoint(
                CHKPT_DIR / scipio_v,
                load_optimizer=False,
                load_scheduler=False,
            )

            hannibal_elo = elo_ratings.get(hannibal.name, ELO_INITIAL)
            scipio_elo = elo_ratings.get(scipio_v, ELO_INITIAL)
            logger.info(
                f"Round {round_num}/{ROUNDS} hannibal vs {scipio_v} "
                f"(ELO: {hannibal_elo:.0f} vs {scipio_elo:.0f})"
            )
            results = run_game(sc2.maps.get(map), [hannibal, scipio], realtime=False, game_time_limit=TIMEOUT)
            result = results[0] if isinstance(results, list) else results

            result = 1 if result == Result.Victory else (0.5 if result == Result.Tie else 0)

            _update_recent_windows(recent_results, result)
            _update_recent_windows(recent_selfplay_results, result)

            elo_ratings[hannibal.name] = update_elo(hannibal_elo, scipio_elo, result)
            elo_ratings[scipio_v] = update_elo(scipio_elo, hannibal_elo, 1 - result)

            logger.info(f"Result: {result}")

            if result == 1:
                stats["wins"] += 1
            elif result == 0:
                stats["losses"] += 1
            else:
                stats["ties"] += 1


        elif opponent_type == "scripted":
            race, bot_class = random.choice(SCRIPTED_BOTS)
            opponent_name = f"{race.name}_{bot_class.__name__}"
            opponent = Bot(race, bot_class(), name=opponent_name)

            hannibal_elo = elo_ratings.get(hannibal.name, ELO_INITIAL)
            scripted_elo = elo_ratings.get(opponent_name, ELO_INITIAL)
            logger.info(
                f"Round {round_num}/{ROUNDS} hannibal vs {opponent_name} "
                f"(ELO: {hannibal_elo:.0f} vs {scripted_elo:.0f})"
            )
            results = run_game(sc2.maps.get(map), [hannibal, opponent], realtime=False, game_time_limit=TIMEOUT)
            result = results[0] if isinstance(results, list) else results

            result = 1 if result == Result.Victory else (0.5 if result == Result.Tie else 0)

            _update_recent_windows(recent_results, result)

            elo_ratings[hannibal.name] = update_elo(hannibal_elo, scripted_elo, result)
            elo_ratings[opponent_name] = update_elo(scripted_elo, hannibal_elo, 1-result)

            logger.info(f"Result: {result}")

            if result == 1:
                stats["wins"] += 1
            elif result == 0:
                stats["losses"] += 1
            else:
                stats["ties"] += 1

        else:
            opponent = MEAT_WALLS[meat_level]
            opponent_name = MEAT_WALLS_NAME[meat_level]

            hannibal_elo = elo_ratings.get(hannibal.name, ELO_INITIAL)
            meatwall_elo = elo_ratings.get(opponent_name, ELO_INITIAL)
            logger.info(
                f"Round {round_num}/{ROUNDS} hannibal vs {opponent_name} "
                f"(ELO: {hannibal_elo:.0f} vs {meatwall_elo:.0f})"
            )
            results = run_game(sc2.maps.get(map), [hannibal, opponent], realtime=False, game_time_limit=TIMEOUT)
            result = results[0] if isinstance(results, list) else results

            result = 1 if result == Result.Victory else (0.5 if result == Result.Tie else 0)

            _update_recent_windows(recent_results, result)

            elo_ratings[hannibal.name] = update_elo(hannibal_elo, meatwall_elo, result)
            elo_ratings[opponent_name] = update_elo(meatwall_elo, hannibal_elo, 1-result)

            logger.info(f"Result: {result}")

            if result == 1:
                stats["wins"] += 1
                meat_level = min(meat_level + 1, len(MEAT_WALLS) - 1)
            elif result == 0:
                stats["losses"] += 1
                meat_level = max(meat_level - 1, 0)
            else:
                stats["ties"] += 1

        metrics = hannibal_agent.get_performance_metrics()
        scores = evaluate_agent(metrics)
        logger.info(f"Performance: {scores}")
        plot_radar_chart(scores, f"charts/performance_round_{round_num}.png")
        plt.close()

        total = stats["wins"] + stats["losses"] + stats["ties"]
        if total > 0:
            win_rate = stats["wins"] / total
            loss_rate = stats["losses"] / total
            tie_rate = stats["ties"] / total
        else:
            win_rate = loss_rate = tie_rate = 0.0

        hannibal_elo = elo_ratings.get(hannibal.name, ELO_INITIAL)

        mlflow.log_metric("game/win_rate", win_rate, step=round_num)
        mlflow.log_metric("game/loss_rate", loss_rate, step=round_num)
        mlflow.log_metric("game/tie_rate", tie_rate, step=round_num)
        mlflow.log_metric("game/elo_rating", hannibal_elo, step=round_num)
        _log_recent_winrates("game", recent_results, round_num)
        _log_recent_winrates("game", recent_selfplay_results, round_num, label="selfplay")

        mlflow.log_metric("stats/total_wins", stats["wins"], step=round_num)
        mlflow.log_metric("stats/total_losses", stats["losses"], step=round_num)
        mlflow.log_metric("stats/total_ties", stats["ties"], step=round_num)

        eco = metrics["economic"]
        mlflow.log_metric("economic/mineral_collection_efficiency", eco["mineral_collection_efficiency"], step=round_num)
        mlflow.log_metric("economic/vespene_collection_efficiency", eco["vespene_collection_efficiency"], step=round_num)
        mlflow.log_metric("economic/idle_worker_time", eco["idle_worker_time"], step=round_num)
        mlflow.log_metric("economic/idle_production_time", eco["idle_production_time"], step=round_num)

        mil = metrics["military"]
        mlflow.log_metric("military/total_damage_dealt", mil["total_damage_dealt"], step=round_num)
        mlflow.log_metric("military/total_damage_taken", mil["total_damage_taken"], step=round_num)
        mlflow.log_metric("military/damage_ratio", mil["damage_ratio"], step=round_num)
        mlflow.log_metric("military/kill_value_ratio", mil["kill_value_ratio"], step=round_num)

        res = metrics["resources"]
        mlflow.log_metric("resources/total_resources_collected", res["total_resources_collected"], step=round_num)
        mlflow.log_metric("resources/total_resources_spent", res["total_resources_spent"], step=round_num)
        mlflow.log_metric("resources/resource_spending_rate", res["resource_spending_rate"], step=round_num)

        prod = metrics["production"]
        mlflow.log_metric("production/total_unit_value", prod["total_unit_value"], step=round_num)
        mlflow.log_metric("production/total_structure_value", prod["total_structure_value"], step=round_num)
        mlflow.log_metric("production/total_value_created", prod["total_value_created"], step=round_num)
        mlflow.log_metric("production/value_lost_units", prod["value_lost_units"], step=round_num)
        mlflow.log_metric("production/value_lost_structures", prod["value_lost_structures"], step=round_num)
        mlflow.log_metric("production/net_value_retained", prod["net_value_retained"], step=round_num)

        comp = metrics["composition"]
        mlflow.log_metric("composition/workers", comp["workers"], step=round_num)
        mlflow.log_metric("composition/army_count", comp["army_count"], step=round_num)
        mlflow.log_metric("composition/structure_count", comp["structure_count"], step=round_num)
        mlflow.log_metric("composition/supply_army", comp["supply_army"], step=round_num)
        mlflow.log_metric("composition/supply_workers", comp["supply_workers"], step=round_num)
        mlflow.log_metric("composition/supply_economy", comp["supply_economy"], step=round_num)
        mlflow.log_metric("composition/supply_technology", comp["supply_technology"], step=round_num)

        mlflow.log_metric("game/time", metrics["game_time"], step=round_num)
        mlflow.log_metric("training/round", round_num)
        mlflow.log_metric("training/meat_level", meat_level, step=round_num)
        for key, value in metrics.get("cumulative", {}).items():
            mlflow.log_metric(f"cumulative/{key}", value, step=round_num)
            
        selfplay_recent = recent_selfplay_results.get(SELFPLAY_SAVE_WINDOW)
        if round_num % SAVE_EPISODES == 0 and selfplay_recent and len(selfplay_recent) > 0:
            recent_selfplay_winrate = _win_rate_from_results(selfplay_recent)

            # Phase expansion logic
            current_phase = hannibal_agent._phase
            phase1_target_round = 50  # Target to beat Easy consistently
            phase2_target_round = 150  # Target to beat Medium consistently
            phase_expand_threshold = 0.7  # 70% win rate to trigger phase change

            if current_phase == 1 and round_num >= phase1_target_round and recent_selfplay_winrate >= phase_expand_threshold:
                logger.info(f"Phase 1 → 2 at round {round_num} (win rate: {recent_selfplay_winrate:.2%})")
                hannibal_agent._phase = 2
                hannibal_agent.expand_action_head(15)  # 15 actions for phase 2
                # Freeze action head for first 300 rounds of new phase
                for param in hannibal_agent.action_head.parameters():
                    param.requires_grad = False

            elif current_phase == 2 and round_num >= phase2_target_round and recent_selfplay_winrate >= phase_expand_threshold:
                logger.info(f"Phase 2 → 3 at round {round_num} (win rate: {recent_selfplay_winrate:.2%})")
                hannibal_agent._phase = 3
                hannibal_agent.expand_action_head(41)  # Full 41 actions
                # Unfreeze action head
                for param in hannibal_agent.action_head.parameters():
                    param.requires_grad = True

            v += 1
            current_phase = hannibal_agent._phase
            new_checkpoint = CHKPT_DIR / f"hannibal_p{current_phase}_v{v}.pt"
            hannibal_agent.save_checkpoint(new_checkpoint)
            prev_elo = elo_ratings.get(hannibal.name, ELO_INITIAL)
            elo_ratings[new_checkpoint.name] = prev_elo
            versions.append(new_checkpoint.name)
            versions.sort(key=lambda key: elo_ratings.get(key, ELO_INITIAL), reverse=True)
            top_versions = versions[:3]
            old_versions = [ver for ver in versions if ver not in top_versions]
            diversity_pick = random.sample(old_versions, min(2, len(old_versions))) if old_versions else []
            versions = top_versions + diversity_pick
            mlflow.log_metric("checkpoint_version", v, round_num)
            mlflow.log_metric("game/recent_selfplay_winrate", recent_selfplay_winrate, step=round_num)
            mlflow.log_metric("training/phase", current_phase, step=round_num)
            log_elo_artifact(elo_ratings)
    
    hannibal_agent.save_checkpoint(CHKPT_DIR / f"hannibal_p{hannibal_agent._phase}_final.pt")

    if mlflow.active_run() is not None:
        mlflow.end_run()


if __name__ == "__main__":
    main()
