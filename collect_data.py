import os

import sc2
from sc2.main import run_game
from sc2.data import Difficulty, Race, Result
from sc2.player import Bot, Computer
from winning_bot import WinningZergBot
from numpy import random

N_GAMES = 7
MAP = "AbyssalReefAIE"

for game in range(1, N_GAMES + 1):
    print(f"\n=== game {game}/{N_GAMES} ===")
    ai = WinningZergBot(
                  max_drones=15,
                  attack_threshold=15,
                  action_interval=4,
                  dataset_path=f"datasets/winning_{game}.pt",
                  collect_data=True,
              )
    ai._log_level = 0
    bot = Bot(Race.Zerg, ai)
    computer = Computer(race = Race.Random, difficulty = Difficulty.VeryEasy)
    result = run_game(
        sc2.maps.get(MAP),
        [bot, computer],
        realtime=False,
    )
    print(f"Game result: {result}")
    if result == Result.Defeat:
        os.remove(f"datasets/winning_{game}.pt")
        N_GAMES += 1