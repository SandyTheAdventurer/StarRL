import sc2
from sc2.main import run_game
from sc2.data import Difficulty
from sc2.data import Race
from sc2.player import Bot, Computer
from winning_bot import WinningZergBot
from staragent import StarAgent

MAP = "AbyssalReefAIE"
EPISODES = 1
difficulty = Difficulty.VeryEasy
RENDER = True
LOG = True

star_bot = Bot(Race.Zerg, StarAgent(train_mode=True, checkpoint_path="checkpoints/bc_agent.pt"))
winner_bot = Bot(
    Race.Zerg,
    WinningZergBot(
        max_drones=16,
        attack_threshold=7,
        action_interval=4,
        collect_data=False,
    ),
)

if not LOG:
    star_bot.ai._log_level = 0
    winner_bot.ai._log_level = 0

computer_bot = Computer(Race.Zerg, difficulty)

for episode in range(1, EPISODES + 1):
    print(f"\n=== Episode {episode}/{EPISODES} ===")
    result = run_game(
        sc2.maps.get(MAP),
        [star_bot, computer_bot],
        realtime=False,
        rgb_render_config={
    "window_size": (1024, 720),
    "minimap_size": (200, 200),
} if RENDER else None,
    )
    print(f"Episode result: {result}")