import sys
import sc2
from sc2.main import run_game
from sc2.data import Difficulty, Race, Result
from sc2.player import Bot, Computer
from scaffold import Scaffold


class TestAllActionsBot(Scaffold):
    def __init__(self):
        super().__init__()
        self._action_results = {}
        self._results_ready = False

    async def on_step(self, iteration: int):
        if iteration > 0 and self._results_ready:
            return

        if iteration % 5 != 0:
            return

        if iteration == 0:
            self._action_results = {}
            for action_idx in range(self.total_actions):
                try:
                    success = await self._execute_action(action_idx)
                    self._action_results[action_idx] = ("ok", success)
                except Exception as e:
                    self._action_results[action_idx] = ("error", str(e))

            self._results_ready = True
            failed = [(idx, msg) for idx, (status, msg) in self._action_results.items() if status == "error"]
            print(f"\n=== Iteration {iteration}: tested {self.total_actions} actions ===")
            if failed:
                print(f"FAILED: {failed}")
                sys.exit(1)
            else:
                print("All actions OK")


def main():
    bot = TestAllActionsBot()

    result = run_game(
        sc2.maps.get("CeruleanFallLE"),
        [Bot(race=Race.Zerg, ai=bot), Computer(Race.Random, Difficulty.Easy)],
        realtime=False,
        game_time_limit=5 * 60,
    )

    print(f"\nGame result: {result}")
    failed = [(idx, msg) for idx, (status, msg) in bot._action_results.items() if status == "error"]
    if failed:
        print(f"Failed actions: {failed}")
        for idx, msg in failed:
            print(f"  Action {idx}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()