import json
import logging
import random
import signal
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from utils import update_elo, evaluate_agent, plot_radar_chart
import sc2
from sc2.data import Difficulty, Race, Result
from sc2.main import run_game
from sc2.player import Bot, Computer

from bots.protoss.cannon_rush import CannonRushBot
from bots.protoss.find_adept_shades import FindAdeptShadesBot
from bots.protoss.threebase_voidray import ThreebaseVoidrayBot
from bots.protoss.warpgate_push import WarpGateBot
from bots.terran.cyclone_push import CyclonePush
from bots.terran.mass_reaper import MassReaperBot
from bots.terran.onebase_battlecruiser import BCRushBot
from bots.terran.proxy_rax import ProxyRaxBot
from bots.zerg.banes_banes_banes import BanesBanesBanes
from bots.zerg.expand_everywhere import ExpandEverywhere
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("self_play")
if logger.handlers:
    logger.handlers[0].setFormatter(ColoredFormatter())
else:
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logger.addHandler(handler)


TIMEOUT = 30 * 60
SAVE_EPISODES = 4
ROUNDS = 200
CHKPT_DIR = Path("checkpoints")
MAPS_DIR = Path("StarCraftII/maps")
ELO_RATINGS_PATH = Path("elo_ratings.json")
EXPERIMENT_NAME = "starcraft-selfplay"

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
    (Race.Zerg, BanesBanesBanes),
    (Race.Zerg, ExpandEverywhere),
    (Race.Zerg, Hydralisk),
    (Race.Zerg, BroodlordBot),
    (Race.Zerg, ZergRushBot),
    (Race.Protoss, CannonRushBot),
    (Race.Protoss, FindAdeptShadesBot),
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
    elo_ratings = {}
    recent_results = []
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

    hannibal_agent = StarAgent(train_mode=True, name="Hannibal", log_mlflow=True)
    hannibal = Bot(Race.Zerg, hannibal_agent, name="Hannibal")
    initial_checkpoint = CHKPT_DIR / f"hannibal_v{v}.pt"
    hannibal_agent.save_checkpoint(initial_checkpoint)
    versions.append(initial_checkpoint.name)
    elo_ratings[hannibal.name] = ELO_INITIAL
    elo_ratings[initial_checkpoint.name] = ELO_INITIAL

    scipio_agent = StarAgent(train_mode=False, name="Scipio", log_mlflow=False)
    scipio = Bot(Race.Zerg, scipio_agent, name="Scipio")

    for round_num in range(1, ROUNDS + 1):
        if should_stop:
            logger.info("Stopping after current round completes.")
            break

        pool_size = len(versions)
        selfplay_weight = min(0.7, 0.3 + 0.04 * pool_size)
        opponent_type = random.choices(
            ["scipio", "scripted", "meatwall"],
            weights=[selfplay_weight, 1 - selfplay_weight - 0.2, 0.2],
        )[0]

        map = map_rng.choice(maps)

        if opponent_type == "scipio":
            scipio_v = random.choice(versions)
            scipio_agent.load_checkpoint(CHKPT_DIR / scipio_v)

            hannibal_elo = elo_ratings.get(hannibal.name, ELO_INITIAL)
            scipio_elo = elo_ratings.get(scipio_v, ELO_INITIAL)
            logger.info(
                f"Round {round_num}/{ROUNDS} hannibal vs {scipio_v} "
                f"(ELO: {hannibal_elo:.0f} vs {scipio_elo:.0f})"
            )
            results = run_game(sc2.maps.get(map), [hannibal, scipio], realtime=False, game_time_limit=TIMEOUT)
            result = results[0] if isinstance(results, list) else results

            result = 1 if result == Result.Victory else (0.5 if result == Result.Tie else 0)

            recent_results.append(result)
            if len(recent_results) > 10:
                recent_results.pop(0)

            elo_ratings[hannibal.name] = update_elo(hannibal_elo, scipio_elo, result)

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

            recent_results.append(result)
            if len(recent_results) > 10:
                recent_results.pop(0)

            elo_ratings[hannibal.name] = update_elo(hannibal_elo, scripted_elo, result)
            elo_ratings[opponent_name] = update_elo(scripted_elo, hannibal_elo, result)

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

            recent_results.append(result)
            if len(recent_results) > 10:
                recent_results.pop(0)

            elo_ratings[hannibal.name] = update_elo(hannibal_elo, meatwall_elo, result)
            elo_ratings[opponent_name] = update_elo(meatwall_elo, hannibal_elo, result)

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

        for key, value in metrics.get("cumulative", {}).items():
            mlflow.log_metric(f"cumulative/{key}", value, step=round_num)
            
        if round_num % SAVE_EPISODES == 0 and len(recent_results) > 0:
            recent_winrate = sum(1 for r in recent_results if r == 1) / len(recent_results)
            if recent_winrate >= 0.5:
                v += 1
                new_checkpoint = CHKPT_DIR / f"hannibal_v{v}.pt"
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
                log_elo_artifact(elo_ratings)
    
    hannibal_agent.save_checkpoint(CHKPT_DIR / f"hannibal_final.pt")

    if mlflow.active_run() is not None:
        mlflow.end_run()


if __name__ == "__main__":
    main()
