"""
Self-play training loop for StarCraft II agents.

Features:
- N episodes of self-play between two agents
- Every EVAL_INTERVAL episodes, the stronger agent faces the built-in AI
- Progressive difficulty: if the agent wins, the difficulty escalates
- Full TensorBoard logging (results, ELO, win rates, difficulty level)

Usage:
    python train_selfplay.py --episodes 500 --eval_interval 20
"""

import argparse
import time
import random
import logging
from collections import deque
from pathlib import Path

import sc2
from sc2.main import run_game
from sc2.data import Race, Difficulty, AIBuild
from sc2.player import Bot, Computer
from staragent import StarAgent
from zerg_bots import CheeseZergBot, ReactiveZergBot, HarassZergBot
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAP = "AbyssalReefAIE"
RENDER = False  # flip to True locally to watch games

DIFFICULTIES = [
    Difficulty.VeryEasy,
    Difficulty.Easy,
    Difficulty.Medium,
    Difficulty.MediumHard,
    Difficulty.Hard,
    Difficulty.Harder,
    Difficulty.VeryHard,
    Difficulty.CheatVision,
    Difficulty.CheatMoney,
    Difficulty.CheatInsane,
]

DIFFICULTY_NAMES = [
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

GENERAL_AGENTS = [
    "Napoleon",
    "Hannibal",
    "Odysseus",
    "SunShinYi",
    "Alexander",
]
GENERAL_BOTS = [
    "CheeseZerg",
    "ReactiveZerg",
    "HarassZerg",
]
GENERAL_COMPUTERS = [
    "Computer_VeryEasy",
    "Computer_Easy",
    "Computer_Medium",
    "Computer_MediumHard"
]
GENERALS = GENERAL_AGENTS + GENERAL_BOTS + GENERAL_COMPUTERS
WINS_TO_PROMOTE = 3
SAVEDIR = "checkpoints"
LOGDIR = "runs/selfplay"
TOTAL_EPISODES = 200
TIMEOUT = 15 * 60
# ---------------------------------------------------------------------------
# ELO helpers
# ---------------------------------------------------------------------------


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_a: float, rating_b: float, score_a: float, k: float = 32):
    """Return (new_a, new_b). score_a is 1 win, 0.5 draw, 0 loss."""
    ea = expected_score(rating_a, rating_b)
    eb = expected_score(rating_b, rating_a)
    new_a = rating_a + k * (score_a - ea)
    new_b = rating_b + k * ((1 - score_a) - eb)
    return new_a, new_b


def make_render_cfg():
    return (
        {
            "window_size": (1024, 720),
            "minimap_size": (200, 200),
        }
        if RENDER
        else None
    )


def build_agent(checkpoint: str) -> StarAgent:
    agent = StarAgent(checkpoint_path=checkpoint, device="cpu")
    agent._log_level = 0
    return agent



def build_cheese_zerg():
    return CheeseZergBot()

def build_reactive_zerg():
    return ReactiveZergBot()

def build_harass_zerg():
    return HarassZergBot()


def play_selfplay_game(
    agent1: StarAgent, agent2: StarAgent, name1="Agent1", name2="Agent2"
):
    """Run one self-play game and return (result, outcome_for_agent1).
    outcome: 1=win, 0.5=draw, 0=loss
    """
    bot1 = Bot(Race.Zerg, agent1, name=name1)
    bot2 = Bot(Race.Zerg, agent2, name=name2)
    result = run_game(
        sc2.maps.get(MAP),
        [bot1, bot2],
        rgb_render_config=make_render_cfg(),
        realtime=False,
        game_time_limit=TIMEOUT,
    )
    r = result[0] if isinstance(result, (list, tuple)) else result
    outcome = (
        1.0 if str(r) == "Result.Victory" else (0.5 if str(r) == "Result.Tie" else 0.0)
    )
    return r, outcome


def play_vs_computer(
    agent: StarAgent, difficulty: Difficulty, ai_build=AIBuild.RandomBuild
):
    """Run one game vs built-in CAI. Returns (result_str, won: bool)."""
    bot = Bot(Race.Zerg, agent, name="Player")
    computer = Computer(Race.Random, difficulty, ai_build)
    result = run_game(
        sc2.maps.get(MAP),
        [bot, computer],
        rgb_render_config=make_render_cfg(),
        realtime=False,
    )
    r = result[0] if isinstance(result, (list, tuple)) else result
    won = str(r) == "Result.Victory"
    return str(r), won


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

writer = SummaryWriter(log_dir=LOGDIR)


class PortpickerFilter(logging.Filter):
    def filter(self, record):
        return (
            "Returning a port that wasn't given by portpicker"
            not in record.getMessage()
        )


logging.getLogger().addFilter(PortpickerFilter())
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("selfplay")

Path(LOGDIR).mkdir(parents=True, exist_ok=True)


generals = {}
elo = {}
wins = {}
for name in GENERALS:
    if name in GENERAL_BOTS:
        if name == "CheeseZerg":
            generals[name] = build_cheese_zerg()
        elif name == "ReactiveZerg":
            generals[name] = build_reactive_zerg()
        elif name == "HarassZerg":
            generals[name] = build_harass_zerg()
    elif name in GENERAL_AGENTS:
        generals[name] = build_agent(f"{SAVEDIR}/{name}.pt")
    elif name.startswith("Computer_"):
        # Store as a tuple (difficulty, name) for later use
        diff_idx = GENERAL_COMPUTERS.index(name)
        difficulty = DIFFICULTIES[diff_idx]
        generals[name] = (difficulty, name)
    elo[name] = 1200.0
    wins[name] = 0
draws = 0

# Difficulty tracking for eval
difficulty_idx = 0
consecutive_wins = 0
recent_selfplay = deque(maxlen=50)

log.info("Starting self-play training for %d episodes", TOTAL_EPISODES)

for episode in range(1, TOTAL_EPISODES + 1):
    ep_start = time.time()
    best_agent = None
    log.info("Starting episode %d/%d", episode, TOTAL_EPISODES)
    gn1, gn2 = random.sample(GENERALS, 2)
    log.info(
        "Self-play match: %s (ELO=%.0f) vs %s (ELO=%.0f)", gn1, elo[gn1], gn2, elo[gn2]
    )

    ent1, ent2 = generals[gn1], generals[gn2]
    # If either is a Computer AI, set up the match accordingly
    if isinstance(ent1, tuple) and isinstance(ent2, tuple):
        # Two computers: skip (no point in running)
        log.info("Skipping Computer vs Computer match: %s vs %s", gn1, gn2)
        continue
    elif isinstance(ent1, tuple):
        # ent1 is Computer, ent2 is agent/bot
        difficulty, cname = ent1
        result, won = play_vs_computer(ent2, difficulty)
        outcome1 = 1.0 if not won else 0.0  # Computer is always player 1
        outcome2 = 1.0 if won else 0.0
    elif isinstance(ent2, tuple):
        # ent2 is Computer, ent1 is agent/bot
        difficulty, cname = ent2
        result, won = play_vs_computer(ent1, difficulty)
        outcome1 = 1.0 if won else 0.0
        outcome2 = 1.0 if not won else 0.0
    else:
        # Both are agents/bots
        result, outcome1 = play_selfplay_game(ent1, ent2, name1=gn1, name2=gn2)
        outcome2 = 1.0 - outcome1

    elo[gn1], elo[gn2] = update_elo(elo[gn1], elo[gn2], outcome1)

    # Update win counts
    if outcome1 == 1.0:
        wins[gn1] += 1
        recent_selfplay.append(1)
    elif outcome1 == 0.0:
        wins[gn2] += 1
        recent_selfplay.append(0)
    else:
        draws += 1
        recent_selfplay.append(0.5)
    ep_time = time.time() - ep_start

    # ---- TensorBoard: self-play metrics for all generals ----
    for name in GENERALS:
        writer.add_scalar(f"elo/{name}", elo[name], episode)
        writer.add_scalar(f"wins/{name}", wins[name], episode)

    writer.add_scalar("draw_rate", draws / episode, episode)
    writer.add_scalar(
        "rolling_win_rate_g1",
        sum(1 for x in recent_selfplay if x == 1) / max(len(recent_selfplay), 1),
        episode,
    )
    writer.add_scalar("episode_time_s", ep_time, episode)

    log.info(
        "Episode %4d/%d | %s vs %s | result=%s | ELO(%s=%.0f, %s=%.0f) | time=%.1fs",
        episode,
        TOTAL_EPISODES,
        gn1,
        gn2,
        result,
        gn1,
        elo[gn1],
        gn2,
        elo[gn2],
        ep_time,
    )

    # ---- League Table Logging ----
    league_table = []
    for name in GENERALS:
        total_games = sum(wins.values())
        wr = wins[name] / total_games * 100 if total_games else 0.0
        league_table.append((name, elo[name], wins[name], wr))
    league_table.sort(key=lambda x: x[1], reverse=True)

    log.info("\n===== League Table (ELO, Wins, WinRate) =====")
    table_str = "Rank | Name       | ELO    | Wins | WinRate\n"
    table_str += "-----+------------+--------+------|--------\n"
    for idx, (name, e, w, wr) in enumerate(league_table, 1):
        table_str += f"{idx:>4} | {name:<10} | {e:>6.0f} | {w:>4} | {wr:5.1f}%\n"
    log.info("\n" + table_str)
    writer.add_text("league_table", table_str, episode)


    # Only StarAgents have save_checkpoint
    if gn1 in GENERAL_AGENTS and hasattr(generals[gn1], "save_checkpoint"):
        generals[gn1].save_checkpoint(f"{SAVEDIR}/{gn1}.pt")
    if gn2 in GENERAL_AGENTS and hasattr(generals[gn2], "save_checkpoint"):
        generals[gn2].save_checkpoint(f"{SAVEDIR}/{gn2}.pt")

# ---- Final summary ----
writer.add_hparams(
    {
        "total_episodes": TOTAL_EPISODES,
        "wins_to_promote": WINS_TO_PROMOTE,
    },
    {
        **{f"hparam/final_elo_{name}": elo[name] for name in GENERALS},
        "hparam/final_difficulty_level": difficulty_idx,
        "hparam/total_draws": draws,
    },
)
writer.close()

log.info("Training complete.")
for name in GENERALS:
    log.info("Final ELO  →  %s=%.0f", name, elo[name])
log.info("Final difficulty reached: %s", DIFFICULTY_NAMES[difficulty_idx])
