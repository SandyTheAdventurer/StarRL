
import sc2
from sc2.main import run_game
from sc2.data import Difficulty, Race, Result
from sc2.player import Bot, Computer
from bots.zerg.onebase_broodlord import BroodlordBot as ZergBot
from staragent import StarAgent

MAP = "CeruleanFallLE"
EPISODES = 1
difficulty = Difficulty.Hard
RENDER = True
TIMEOUT = 30 * 60  # 30 minutes

ai = StarAgent(train_mode=False, log_mlflow=False, compile_model=True)
ai.load_checkpoint("checkpoints/hannibal_p1_v10.pt")
zerg_ai = ZergBot()
star_bot = Bot(
    Race.Zerg, ai
)
zerg_bot = Bot(
    Race.Zerg, zerg_ai
)

bot2 = Computer(Race.Random, difficulty)
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