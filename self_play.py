import sc2
from sc2.main import run_game
from sc2.data import Difficulty, Race, Result
from sc2.player import Bot, Computer
from staragent import StarAgent
import mlflow
import numpy as np
import asyncio
import logging
import signal
import os

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
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.RESET)
        record.levelname = f"{color}{levelname}{self.RESET}"
        return super().format(record)


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

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

TIMEOUT = 15 * 60
EVAL_EPISODES = 10
SAVE_EPISODES = 3
PROMOTION_TRIES = 3
ROUNDS = 200
CHKPT_DIR = "checkpoints"
MAPS = []

for root, dirs, files in os.walk("StarCraftII/maps"):
        for file in files:
            if file.endswith(".SC2Map"):
                map_name = os.path.splitext(file)[0]
                MAPS.append(map_name)

print(f"Found {len(MAPS)}")

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
    Computer(Race.Random, Difficulty.CheatInsane)
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
    "CheatInsane"
]

meat_level = 0
v = 0
versions = []
wins = 0
games_played = 0

hannibal_agent = StarAgent(train_mode=True, name="Hannibal", log_mlflow=True)
hannibal = Bot(Race.Zerg, hannibal_agent, name="Hannibal")
hannibal_agent.save_checkpoint(f"{CHKPT_DIR}/hannibal_v{v}.pt")
versions.append(f"hannibal_v{v}.pt")

scipio_agent = StarAgent(train_mode=False, name="Scipio", log_mlflow=False)
scipio = Bot(Race.Zerg, scipio_agent, name="Scipio")

mlflow.set_experiment("starcraft-selfplay")
mlflow.end_run()
mlflow.start_run()

should_stop = False


def signal_handler(signum, frame):
    global should_stop
    logger.warning("Received Ctrl+C, finishing current round then stopping...")
    should_stop = True


signal.signal(signal.SIGINT, signal_handler)

for round in range(1, ROUNDS + 1):
    if should_stop:
        logger.info("Stopping after current round completes.")
        break

    if round % EVAL_EPISODES != 0:
        scipio_v = np.random.choice(versions)
        scipio_agent.load_checkpoint(f"{CHKPT_DIR}/{scipio_v}")
        logger.info(f"Round {round}/{ROUNDS} hannibal vs {scipio_v}")
        results = run_game(sc2.maps.get(np.random.choice(MAPS)), [hannibal, scipio], realtime=False, game_time_limit=TIMEOUT)
        logger.debug(f"Results: {results}")
        hannibal_result = (results[0] if isinstance(results, list) else results) == Result.Victory
        hannibal_tie = (results[0] if isinstance(results, list) else results) == Result.Tie
        logger.info(f"Round {round} result: {'win' if hannibal_result else 'tie' if hannibal_tie else 'loss'}")
        if hannibal_result:
            wins += 1
        games_played += 1
        mlflow.log_metric("win", int(hannibal_result))
        mlflow.log_metric("win_rate", wins / games_played if games_played > 0 else 0)
    else:
        opponent = MEAT_WALLS[meat_level]
        opponent_name = MEAT_WALLS_NAME[meat_level]
        logger.info(f"Round {round}/{ROUNDS} hannibal vs {opponent_name}")
        for i in range(PROMOTION_TRIES):
            results = run_game(sc2.maps.get(np.random.choice(MAPS)), [hannibal, opponent], realtime=False, game_time_limit=TIMEOUT)
            logger.debug(f"Promotion results: {results}")
            hannibal_result = (results[0] if isinstance(results, list) else results) == Result.Victory
            hannibal_tie = (results[0] if isinstance(results, list) else results) == Result.Tie
            logger.info(f"Round {round} Tryout {i+1}: {'win' if hannibal_result else 'tie' if hannibal_tie else 'loss'}")
            mlflow.log_metric("promotion_try", int(hannibal_result), step=i)
            if hannibal_result:
                wins += 1
                games_played += 1
                break
        else:
            meat_level = min(meat_level + 1, len(MEAT_WALLS) - 1)
            logger.info(f"Promoting to next meat wall: {MEAT_WALLS_NAME[meat_level]}")
        mlflow.log_metric("meat_level", meat_level)

    if round % SAVE_EPISODES == 0:
        v += 1
        hannibal_agent.save_checkpoint(f"{CHKPT_DIR}/hannibal_v{v}.pt")
        versions.append(f"hannibal_v{v}.pt")
        mlflow.log_metric("checkpoint_version", v)

    mlflow.log_metric("round", round)

logger.info(f"Training complete. Final win rate: {wins/games_played if games_played > 0 else 0:.2%}")
mlflow.end_run()