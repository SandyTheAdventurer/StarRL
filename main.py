
import sc2
from sc2.main import run_game
from sc2.data import Difficulty, Race, Result
from sc2.player import Bot, Computer
from zerg_bots import CheeseZergBot
from staragent import StarAgent

MAP = "CeruleanFallLE"
EPISODES = 1
difficulty = Difficulty.VeryEasy
RENDER = True
TIMEOUT = 15 * 60 * 22.4  # 15 minutes
LOG = False

ai = StarAgent(train_mode=False, log_mlflow=False)
ai.load_checkpoint("checkpoints/hannibal_v6.pt")

star_bot = Bot(
    Race.Zerg, ai
)

bot = Computer(Race.Random, difficulty)

for episode in range(1, EPISODES + 1):
    print(f"\n=== Episode {episode}/{EPISODES} ===")
    result = run_game(
        sc2.maps.get(MAP),
        [star_bot, bot],
        realtime=False,
        rgb_render_config={
            "window_size": (1024, 720),
            "minimap_size": (200, 200),
        }
        if RENDER
        else None,
        game_time_limit=TIMEOUT,
    )
    print(f"Episode {episode} result: {result}")