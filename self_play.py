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

# Wins needed in a row against current difficulty before promoting
WINS_TO_PROMOTE = 3
CHECKPOINT1 = "checkpoints/agent1.pt"
CHECKPOINT2 = "checkpoints/agent2.pt"
SAVEDIR = "checkpoints"
LOGDIR = "runs/selfplay"
TOTAL_EPISODES = 100
EVAL_INTERVAL = 5

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    agent = StarAgent(checkpoint_path=checkpoint)
    agent._log_level = 0
    return agent


def play_selfplay_game(agent1: StarAgent, agent2: StarAgent):
    """Run one self-play game and return (result, outcome_for_agent1).
    outcome: 1=win, 0.5=draw, 0=loss
    """
    bot1 = Bot(Race.Zerg, agent1)
    bot2 = Bot(Race.Zerg, agent2)
    result = run_game(
        sc2.maps.get(MAP),
        [bot1, bot2],
        rgb_render_config=make_render_cfg(),
        realtime=False,
    )
    r = result[0] if isinstance(result, (list, tuple)) else result
    outcome = (
        1.0 if str(r) == "Result.Victory" else (0.5 if str(r) == "Result.Tie" else 0.0)
    )
    return r, outcome


def play_vs_computer(
    agent: StarAgent, difficulty: Difficulty, ai_build=AIBuild.RandomBuild
):
    """Run one game vs built-in AI. Returns (result_str, won: bool)."""
    bot = Bot(Race.Zerg, agent)
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


def strongest_agent(elo1: float, elo2: float, agent1: StarAgent, agent2: StarAgent):
    """Return the agent with the higher ELO and its index (1 or 2)."""
    if elo1 >= elo2:
        return agent1, 1, elo1
    return agent2, 2, elo2


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

writer = SummaryWriter(log_dir=LOGDIR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("selfplay")

Path(LOGDIR).mkdir(parents=True, exist_ok=True)

agent1 = build_agent(CHECKPOINT1)
agent2 = build_agent(CHECKPOINT2)

elo1, elo2 = 1000.0, 1000.0
win1, win2, draws = 0, 0, 0

# Difficulty tracking for eval
difficulty_idx = 0  # index into DIFFICULTIES list
consecutive_wins = 0  # wins in a row at current difficulty
recent_selfplay = deque(maxlen=50)  # last-N outcomes for win-rate smoothing

log.info("Starting self-play training for %d episodes", TOTAL_EPISODES)
log.info(
    "Eval every %d episodes | %d wins to promote difficulty",
    EVAL_INTERVAL,
    WINS_TO_PROMOTE,
)

for episode in range(1, TOTAL_EPISODES + 1):
    ep_start = time.time()

    # ---- Self-play game -----------------------------------------------
    result, outcome1 = play_selfplay_game(agent1, agent2)
    outcome2 = 1.0 - outcome1

    elo1, elo2 = update_elo(elo1, elo2, outcome1)

    if outcome1 == 1.0:
        win1 += 1
        recent_selfplay.append(1)
    elif outcome1 == 0.0:
        win2 += 1
        recent_selfplay.append(0)
    else:
        draws += 1
        recent_selfplay.append(0.5)
    ep_time = time.time() - ep_start

    # ---- TensorBoard: self-play metrics --------------------------------
    writer.add_scalar("selfplay/elo_agent1", elo1, episode)
    writer.add_scalar("selfplay/elo_agent2", elo2, episode)
    writer.add_scalar("selfplay/elo_delta", abs(elo1 - elo2), episode)
    writer.add_scalar("selfplay/win_rate_agent1", win1 / episode, episode)
    writer.add_scalar("selfplay/win_rate_agent2", win2 / episode, episode)
    writer.add_scalar("selfplay/draw_rate", draws / episode, episode)
    writer.add_scalar(
        "selfplay/rolling_win_rate_agent1",
        sum(1 for x in recent_selfplay if x == 1) / max(len(recent_selfplay), 1),
        episode,
    )
    writer.add_scalar("selfplay/episode_time_s", ep_time, episode)

    log.info(
        "Episode %4d/%d | result=%s | ELO(1=%.0f, 2=%.0f) | time=%.1fs",
        episode,
        TOTAL_EPISODES,
        result,
        elo1,
        elo2,
        ep_time,
    )

    # ---- Periodic evaluation vs built-in AI ---------------------------
    if episode % EVAL_INTERVAL == 0:
        best_agent, best_idx, best_elo = strongest_agent(elo1, elo2, agent1, agent2)
        diff = DIFFICULTIES[difficulty_idx]
        diff_name = DIFFICULTY_NAMES[difficulty_idx]

        log.info(
            ">>> EVAL  episode=%d | strongest=agent%d (ELO=%.0f) | difficulty=%s",
            episode,
            best_idx,
            best_elo,
            diff_name,
        )

        eval_start = time.time()
        eval_result, won = play_vs_computer(best_agent, diff)
        eval_time = time.time() - eval_start

        consecutive_wins = (consecutive_wins + 1) if won else 0

        # Promote difficulty?
        promoted = False
        if (
            consecutive_wins >= WINS_TO_PROMOTE
            and difficulty_idx < len(DIFFICULTIES) - 1
        ):
            difficulty_idx += 1
            consecutive_wins = 0
            promoted = True
            log.info(
                ">>> PROMOTED to difficulty=%s after %d consecutive wins!",
                DIFFICULTY_NAMES[difficulty_idx],
                WINS_TO_PROMOTE,
            )

            # TensorBoard: eval metrics
        writer.add_scalar("eval/difficulty_level", difficulty_idx, episode)
        writer.add_scalar("eval/won_vs_computer", int(won), episode)
        writer.add_scalar(
            "eval/consecutive_wins_at_difficulty", consecutive_wins, episode
        )
        writer.add_scalar("eval/promoted", int(promoted), episode)
        writer.add_scalar("eval/eval_time_s", eval_time, episode)
        writer.add_text(
            "eval/summary",
            f"ep={episode} agent=agent{best_idx} ELO={best_elo:.0f} "
            f"difficulty={diff_name} result={eval_result} won={won} "
            f"consec_wins={consecutive_wins} promoted={promoted}",
            episode,
        )

        log.info(
            ">>> EVAL result=%s | won=%s | consec_wins=%d/%d | promoted=%s | time=%.1fs",
            eval_result,
            won,
            consecutive_wins,
            WINS_TO_PROMOTE,
            promoted,
            eval_time,
        )

        # Optionally save the best agent checkpoint after eval
        ckpt_path = Path(LOGDIR) / f"best_agent_ep{episode}.pt"
        if hasattr(best_agent, "_save_checkpoint"):
            best_agent._save_checkpoint(str(ckpt_path))
            log.info("Saved checkpoint: %s", ckpt_path)

    # ---- Final summary -----------------------------------------------------
writer.add_hparams(
    {
        "total_episodes": TOTAL_EPISODES,
        "eval_interval": EVAL_INTERVAL,
        "wins_to_promote": WINS_TO_PROMOTE,
    },
    {
        "hparam/final_elo_agent1": elo1,
        "hparam/final_elo_agent2": elo2,
        "hparam/final_difficulty_level": difficulty_idx,
        "hparam/total_wins_agent1": win1,
        "hparam/total_draws": draws,
    },
)
writer.close()

log.info("Training complete.")
log.info("Final ELO  →  agent1=%.0f  agent2=%.0f", elo1, elo2)
log.info("Final difficulty reached: %s", DIFFICULTY_NAMES[difficulty_idx])
