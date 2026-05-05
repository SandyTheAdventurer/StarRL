import sys

from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer

from bot import CompetitiveBot
from ladder import run_ladder_game

bot = Bot(Race.Zerg, CompetitiveBot())

if __name__ == "__main__":
    if "--LadderServer" in sys.argv:
        print("Starting ladder game...")
        result, opponent_id = run_ladder_game(bot)
        print(result, "against opponent", opponent_id)
    else:
        print("Starting local game...")
        run_game(
            maps.get("Abyssal Reef LE"),
            [bot, Computer(Race.Protoss, Difficulty.VeryHard)],
            realtime=True,
        )
