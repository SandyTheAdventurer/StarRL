from scaffold import Scaffold
from sc2.ids.unit_typeid import UnitTypeId
from utils import Observer
import torch


class WinningZergBot(Scaffold, Observer):
    def __init__(
        self,
        max_drones: int = 16,
        attack_threshold: int = 6,
        action_interval: int = 4,
        dataset_path: str = "datasets/expert_zerg.pt",
        collect_data: bool = True,
        defend_until_time: float = 110.0,
    ):
        super().__init__()
        Observer.__init__(self)
        self.max_drones = max(12, int(max_drones))
        self.attack_threshold = max(6, int(attack_threshold))
        self.rush_ling_goal = 22
        self.action_interval = max(1, int(action_interval))
        self.dataset_path = dataset_path
        self.collect_data = bool(collect_data)
        self.defend_until_time = max(45.0, float(defend_until_time))
        self._attacking = False
        self._last_known_enemy_pos = None

    def _action_to_one_hot(self, action_idx: int) -> torch.Tensor:
        action = torch.zeros(self.total_actions, dtype=torch.float32)
        if 0 <= int(action_idx) < self.total_actions:
            action[int(action_idx)] = 1.0
        return action

    async def _execute_action(self, action_idx: int):
        if action_idx == 0:
            return False
        if action_idx == 1:
            return await self.train_drones(1)
        if action_idx == 2:
            return await self.train_overlord(1)
        if action_idx == 3:
            return await self.build_extractor()
        if action_idx == 4:
            return await self.build_spawning_pool()
        if action_idx == 5:
            return await self.gather_minerals()
        if action_idx == 6:
            return await self.gather_vespene()
        if action_idx == 7:
            return await self.train_zerglings(4)
        if action_idx == 8:
            return await self.attack_move(target=self._get_attack_target())
        if action_idx == 9:
            return await self.rally_army()
        if action_idx == 10:
            return await self.build_baneling_nest()
        if action_idx == 11:
            return await self.train_banelings(2)
        return False

    def _get_attack_target(self):
        if self.enemy_units:
            close_enemies = self.enemy_units.closer_than(35, self.start_location)
            if close_enemies:
                target = close_enemies.closest_to(self.start_location).position
                self._last_known_enemy_pos = target
                return target

            target = self.enemy_units.closest_to(
                self.units.center if self.units else self.start_location
            ).position
            self._last_known_enemy_pos = target
            return target

        if self.enemy_structures:
            target = self.enemy_structures.closest_to(self.start_location).position
            self._last_known_enemy_pos = target
            return target

        if self._last_known_enemy_pos is not None:
            return self._last_known_enemy_pos

        return self.enemy_start_locations[0]

    def _select_expert_action(self, iteration: int) -> int:
        pool_ready = self.structures(UnitTypeId.SPAWNINGPOOL).ready
        pool_pending = self.already_pending(UnitTypeId.SPAWNINGPOOL)
        lings = self.units(UnitTypeId.ZERGLING)
        banelings = self.units(UnitTypeId.BANELING)
        army_count = lings.amount + banelings.amount
        baneling_nest = self.structures(UnitTypeId.BANELINGNEST)
        hatchery = self.townhalls.ready.first if self.townhalls.ready else None
        close_enemies = (
            self.enemy_units.closer_than(30, self.start_location)
            if self.enemy_units
            else None
        )

        if not pool_ready and not pool_pending:
            if self.supply_left <= 2 and not self.already_pending(UnitTypeId.OVERLORD):
                return 2
            if self.can_afford(UnitTypeId.SPAWNINGPOOL):
                return 4
            if self.workers.amount < min(13, self.max_drones) and self.supply_left > 0:
                return 1
            return 5

        if self.supply_left <= 3 and not self.already_pending(UnitTypeId.OVERLORD):
            return 2

        if self.supply_left == 0 and not self.already_pending(UnitTypeId.OVERLORD):
            return 2

        extractors = self.structures(UnitTypeId.EXTRACTOR)
        extractor_pending = self.already_pending(UnitTypeId.EXTRACTOR)
        if (
            pool_ready
            and not extractors
            and not extractor_pending
            and self.workers.amount >= 17
            and not close_enemies
            and self.can_afford(UnitTypeId.EXTRACTOR)
        ):
            return 3

        if (
            pool_ready
            and not baneling_nest
            and not self.already_pending(UnitTypeId.BANELINGNEST)
            and lings.amount >= 8
            and self.workers.amount >= 15
            and self.can_afford(UnitTypeId.BANELINGNEST)
        ):
            return 10

        if (
            baneling_nest.ready
            and close_enemies
            and lings.amount >= 2
            and banelings.amount < 4
            and self.can_afford(UnitTypeId.BANELING)
        ):
            return 11

        if close_enemies:
            if self.supply_left <= 2 and not self.already_pending(UnitTypeId.OVERLORD):
                return 2
            if (
                pool_ready
                and self.supply_left > 0
                and self.can_afford(UnitTypeId.ZERGLING)
            ):
                return 7
            if army_count > 0:
                self._attacking = False
                return 8
            return 5

        if pool_ready and lings.amount < self.rush_ling_goal and self.supply_left > 0:
            if self.can_afford(UnitTypeId.ZERGLING):
                return 7

        if (
            self.workers.amount < self.max_drones
            and self.supply_left > 0
            and not close_enemies
        ):
            return 1

        if (
            self.structures(UnitTypeId.EXTRACTOR).ready
            and self.vespene < 80
            and army_count >= 8
        ):
            return 6

        if close_enemies and army_count > 0:
            self._attacking = False
            return 8

        if hatchery and lings.idle.amount >= 2 and not self._attacking:
            if lings.center.distance_to(hatchery.position) > 16:
                return 9

        retreat_threshold = max(4, self.attack_threshold // 2)
        if self._attacking:
            if army_count < retreat_threshold:
                self._attacking = False
            else:
                return 8

        if (
            self.time > self.defend_until_time
            and army_count >= self.attack_threshold
            and not close_enemies
        ):
            self._attacking = True
            return 8

        if (
            self.time > self.defend_until_time + 30
            and army_count >= max(4, self.attack_threshold - 2)
            and self.enemy_structures
            and not close_enemies
        ):
            self._attacking = True
            return 8

        if army_count > 0:
            return 9

        if pool_ready and self.supply_left > 0 and self.can_afford(UnitTypeId.ZERGLING):
            return 7

        if self.workers.idle and self.mineral_field:
            return 5

        return 0

    async def on_step(self, iteration: int):
        if iteration % 8 == 0:
            await self.distribute_workers()
        if iteration % 4 == 0:
            await self.maintain_worker_economy(max_idle_orders=8)

        await self.ensure_overlord_buffer(supply_buffer=2)

        close_enemies = (
            self.enemy_units.closer_than(30, self.start_location)
            if self.enemy_units
            else None
        )
        if close_enemies and self.units.of_type(
            {UnitTypeId.ZERGLING, UnitTypeId.BANELING}
        ):
            await self.attack_move(
                target=close_enemies.closest_to(self.start_location).position
            )

        log_this_step = self.collect_data and (iteration % self.action_interval == 0)
        if log_this_step:
            obs_img, obs_res = self.get_observation()

        action_idx = self._select_expert_action(iteration)
        action_ok = await self._execute_action(action_idx)

        self._log_step(iteration, action_idx, action_ok, 0.0)

        if log_this_step:
            self.record_step(
                observation={"image": obs_img, "resources": obs_res},
                action_one_hot=self._action_to_one_hot(action_idx),
                reward=0.0,
                done=False,
                info={
                    "iteration": int(iteration),
                    "action_idx": int(action_idx),
                    "label_only": True,
                },
            )

    async def on_end(self, result):
        if self.collect_data and self.done_history:
            self.done_history[-1] = True
            self.append_episode(
                self.dataset_path,
                metadata={
                    "result": getattr(result, "name", str(result)),
                    "map": getattr(self.game_info, "map_name", "unknown"),
                    "max_drones": int(self.max_drones),
                    "attack_threshold": int(self.attack_threshold),
                    "total_actions": int(self.total_actions),
                },
                clear_after_save=True,
            )

        metrics = self._collect_metrics()
        self._log_episode_end(result, metrics)
