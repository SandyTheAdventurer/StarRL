from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
import numpy as np
import torch
import torch.nn.functional as F


class Scaffold(BotAI):
    def __init__(
        self,
        reward_weights: dict | None = None,
        log_level: int = 0,
    ):
        super().__init__()
        self.observation_space = ((12, 64, 64), 7)
        # 0: no-op
        # 1: build spawning pool
        # 2: train zerglings
        # 3: attack
        # 4: train drones
        # 5: train overlord
        # 6: train anti-air (hydralisk)
        # 7: train flying (mutalisk)
        # 8: build hydralisk den
        # 9: build spire
        # 10: build roach warren
        # 11: train roach
        # 12: build baneling nest
        # 13: train baneling
        # 14: build infestation pit
        # 15: build greater spire
        # 16: train brood lord
        # 17: build spine crawler
        # 18: build spore crawler
        # 19: inject larva
        # 20: spread creep
        # 21: transfuse
        # 22: research zergling speed
        # 23: research burrow
        # 24: research roach speed
        # 25: research baneling speed
        # 26: research flyer attacks
        # 27: retreat
        # 28: regroup
        # 29: focus fire
        # 30: morph to lair
        # 31: morph to greater spire
        # 32: send drone to mine minerals
        # 33: send drone to mine gas
        # 34: explore with zergling
        # 35: build extractor
        # 36: expand (build hatchery)
        # 37: gather vespene
        # 38: gather minerals
        # 39: maintain worker economy
        # 40: ensure overlord buffer
        # 41: flood zerglings
        self.total_actions = 43

        self._reward_weights = reward_weights or {
            "minerals": 0.05,
            "gas": 0.05,
            "workers": 0.075,
            "army": 0.01,
            "kills": 0.015,
            "losses": -0.01,
            "worker_loss": -0.01,
            "success": 0.01,
            "supply_penalty": -0.02,
            "mineral_rate": 0.02,
            "gas_rate": 0.02,
            "enemy_structure_damage": 0.001,
            "enemy_structure_destroyed": 0.01,
            "unit_preservation": 0.02,
            "expansion": 0.01,
        }
        self._log_level = log_level
        self.army_units = [UnitTypeId.ZERGLING, UnitTypeId.BANELING, UnitTypeId.HYDRALISK, UnitTypeId.MUTALISK, UnitTypeId.ROACH, UnitTypeId.CORRUPTOR, UnitTypeId.BROODLORD]
        self._prev_metrics = None

    def get_observation(self):
        target_h = self.observation_space[0][1]
        target_w = self.observation_space[0][2]

        height = self._pixel_map_to_array(self.game_info.terrain_height)
        visibility = self._pixel_map_to_array(self.state.visibility)
        creep = self._pixel_map_to_array(self.state.creep)
        pathing = self._pixel_map_to_array(self.game_info.pathing_grid)

        height = height / 255.0
        visibility = visibility / 2.0
        creep = creep.astype(np.float32)
        pathing = pathing.astype(np.float32)

        friendly, enemy = self.get_unit_map()

        spatial_layers = np.stack(
            [
                self._resize_map(height.astype(np.float32), target_h, target_w),
                self._resize_map(visibility.astype(np.float32), target_h, target_w),
                self._resize_map(creep, target_h, target_w),
                self._resize_map(pathing, target_h, target_w),
            ],
            axis=0,
        )

        friendly_layers = self._resize_map_channels(
            friendly.transpose(2, 0, 1), target_h, target_w
        )
        enemy_layers = self._resize_map_channels(
            enemy.transpose(2, 0, 1), target_h, target_w
        )

        img = np.concatenate([spatial_layers, friendly_layers, enemy_layers], axis=0)

        gas_available = min(self.vespene / 1000.0, 1.0)
        minerals_available = min(self.minerals / 1500.0, 1.0)
        supply_used = self.supply_used / 200.0
        supply_cap = self.supply_cap / 200.0
        game_time = min(self.time / 1800.0, 1.0)
        worker_count = self.units(UnitTypeId.DRONE).amount / 100.0
        army_count = (
            max(
                self.units.not_flying.amount - self.units(UnitTypeId.DRONE).amount,
                0,
            )
            / 100.0
        )
        resources = np.array(
            [
                gas_available,
                minerals_available,
                supply_used,
                supply_cap,
                game_time,
                worker_count,
                army_count,
            ],
            dtype=np.float32,
        )

        return img, resources

    @staticmethod
    def _pixel_map_to_array(pixel_map) -> np.ndarray:
        h = int(getattr(pixel_map, "height"))
        w = int(getattr(pixel_map, "width"))

        raw = getattr(pixel_map, "data", None)
        if raw is None:
            raw = getattr(pixel_map, "_data", None)

        if raw is not None and isinstance(raw, (bytes, bytearray, memoryview)):
            bpp = int(getattr(pixel_map, "bits_per_pixel", 8))
            if bpp <= 8:
                dtype = np.uint8
            elif bpp <= 16:
                dtype = np.uint16
            else:
                dtype = np.uint32

            flat = np.frombuffer(raw, dtype=dtype)
            if flat.size == h * w:
                return flat.reshape((h, w)).astype(np.float32)

        try:
            arr = np.asarray(pixel_map, dtype=np.float32)
            if arr.shape == (h, w):
                return arr
        except Exception:
            pass

        try:
            arr = np.zeros((h, w), dtype=np.float32)
            for y in range(h):
                for x in range(w):
                    arr[y, x] = float(pixel_map[x, y])
            return arr
        except Exception:
            pass

        return np.zeros((h, w), dtype=np.float32)

    @staticmethod
    def _resize_map(layer: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if layer.shape == (target_h, target_w):
            return layer.astype(np.float32)
        tensor = torch.from_numpy(layer).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=(target_h, target_w), mode="nearest")
        return resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    @staticmethod
    def _resize_map_channels(
        layer: np.ndarray, target_h: int, target_w: int
    ) -> np.ndarray:
        if layer.shape[1:] == (target_h, target_w):
            return layer.astype(np.float32)
        tensor = torch.from_numpy(layer).unsqueeze(0)
        resized = F.interpolate(tensor, size=(target_h, target_w), mode="nearest")
        return resized.squeeze(0).cpu().numpy().astype(np.float32)

    def get_unit_map(self):
        w, h = map(int, self.game_info.map_size)

        friendly = np.zeros((h, w, 4), dtype=np.float32)
        enemy = np.zeros((h, w, 4), dtype=np.float32)

        for unit in self.units:
            x, y = int(unit.position.x), int(unit.position.y)
            if 0 <= x < w and 0 <= y < h:
                ny = (h - 1) - y
                friendly[ny][x][0] = unit.type_id.value / 2004.0
                friendly[ny][x][1] = unit.health / max(unit.health_max, 1e-6)
                friendly[ny][x][2] = unit.shield / max(unit.shield_max, 1e-6)
                friendly[ny][x][3] = unit.energy / max(unit.energy_max, 1e-6)

        for unit in self.enemy_units:
            x, y = int(unit.position.x), int(unit.position.y)
            if 0 <= x < w and 0 <= y < h:
                ny = (h - 1) - y
                enemy[ny][x][0] = unit.type_id.value / 2004.0
                enemy[ny][x][1] = unit.health / max(unit.health_max, 1e-6)
                enemy[ny][x][2] = unit.shield / max(unit.shield_max, 1e-6)
                enemy[ny][x][3] = unit.energy / max(unit.energy_max, 1e-6)

        return friendly, enemy

    async def expand(self) -> bool:
        if not self.can_afford(UnitTypeId.HATCHERY):
            return False
        await self.expand_now()
        return True

    async def _execute_action(self, action_idx: int):
        # See class docstring for action mapping
        if action_idx == 0:
            return False
        if action_idx == 1:
            return await self.build_spawning_pool()
        if action_idx == 2:
            return await self.train_zerglings(6)
        if action_idx == 3:
            return await self.attack_move()
        if action_idx == 4:
            return await self.train_drones(1)
        if action_idx == 5:
            return await self.train_overlord(1)
        if action_idx == 6:
            return await self.train_anti_air(2)
        if action_idx == 7:
            return await self.train_flying_unit(2)
        if action_idx == 8:
            return await self.build_hydralisk_den()
        if action_idx == 9:
            return await self.build_spire()
        if action_idx == 10:
            return await self.build_roach_warren()
        if action_idx == 11:
            return await self.train_roach(2)
        if action_idx == 12:
            return await self.build_baneling_nest()
        if action_idx == 13:
            return await self.train_banelings(2)
        if action_idx == 14:
            return await self.build_infestation_pit()
        if action_idx == 15:
            return await self.build_greater_spire()
        if action_idx == 16:
            return await self.train_brood_lord(1)
        if action_idx == 17:
            return await self.build_spine_crawler()
        if action_idx == 18:
            return await self.build_spore_crawler()
        if action_idx == 19:
            return await self.inject_larva()
        if action_idx == 20:
            return await self.spread_creep()
        if action_idx == 21:
            return await self.transfuse()
        if action_idx == 22:
            return await self.research_zergling_speed()
        if action_idx == 23:
            return await self.research_burrow()
        if action_idx == 24:
            return await self.research_roach_speed()
        if action_idx == 25:
            return await self.research_baneling_speed()
        if action_idx == 26:
            return await self.research_flyer_attacks()
        if action_idx == 27:
            return await self.retreat()
        if action_idx == 28:
            return await self.regroup()
        if action_idx == 29:
            return await self.focus_fire()
        if action_idx == 30:
            return await self.morph_to_lair()
        if action_idx == 31:
            return await self.morph_to_greater_spire()
        if action_idx == 32:
            return await self.send_drone_mineral()
        if action_idx == 33:
            return await self.send_drone_gas()
        if action_idx == 34:
            return await self.explore()
        if action_idx == 35:
            return await self.build_extractor()
        if action_idx == 36:
            return await self.expand()
        if action_idx == 37:
            return await self.gather_vespene()
        if action_idx == 38:
            return await self.gather_minerals()
        if action_idx == 39:
            return await self.maintain_worker_economy()
        if action_idx == 40:
            return await self.ensure_overlord_buffer()
        if action_idx == 41:
            return await self.flood_zerglings()
        return False

    async def build_hydralisk_den(self):
        if self.can_afford(UnitTypeId.HYDRALISKDEN) and self.structures(UnitTypeId.LAIR).ready:
            lair = self.structures(UnitTypeId.LAIR).ready.first
            pos = lair.position.towards(self.game_info.map_center, 5)
            await self.build(UnitTypeId.HYDRALISKDEN, near=pos)
            return True
        return False

    async def build_spire(self):
        if self.can_afford(UnitTypeId.SPIRE) and self.structures(UnitTypeId.LAIR).ready:
            lair = self.structures(UnitTypeId.LAIR).ready.first
            pos = lair.position.towards(self.game_info.map_center, 5)
            await self.build(UnitTypeId.SPIRE, near=pos)
            return True
        return False

    async def train_anti_air(self, n=1):
        # Example: Hydralisk (anti-air ground unit)
        if self.can_afford(UnitTypeId.HYDRALISK) and self.structures(UnitTypeId.HYDRALISKDEN).ready:
            for _ in range(n):
                larva_candidates = self.units(UnitTypeId.LARVA).filter(
                    lambda l: l.tag not in self.unit_tags_received_action
                )
                if larva_candidates:
                    larva = larva_candidates.first
                    self.do(larva.train(UnitTypeId.HYDRALISK))
                else:
                    return False  # No available larva
            return True
        return False

    async def train_flying_unit(self, n=1):
        # Example: Mutalisk (flying unit)
        if self.can_afford(UnitTypeId.MUTALISK) and self.structures(UnitTypeId.SPIRE).ready:
            for _ in range(n):
                larva = self.units(UnitTypeId.LARVA).filter(lambda l: l.tag not in self.unit_tags_received_action).first
                if larva:
                    self.do(larva.train(UnitTypeId.MUTALISK))
            return True
        return False
    
    async def train_drones(self, n: int = 1) -> bool:
        issued = False
        for _ in range(max(1, n)):
            if (
                self.larva
                and self.can_afford(UnitTypeId.DRONE)
                and self.supply_left > 0
            ):
                if not self.larva:
                    break
                larva = self.larva.random
                self.do(larva.train(UnitTypeId.DRONE))
                issued = True
        return issued

    async def train_zerglings(self, n: int = 1) -> bool:
        if not self.structures(UnitTypeId.SPAWNINGPOOL).ready:
            return False

        issued = False
        for _ in range(max(1, n)):
            if not self.can_afford(UnitTypeId.ZERGLING):
                break
            available_larva = self.larva.filter(lambda l: l.tag not in self.unit_tags_received_action)
            if not available_larva:
                break
            larva = available_larva.random
            self.do(larva.train(UnitTypeId.ZERGLING))
            issued = True
        return issued

    async def train_banelings(self, n: int = 1) -> bool:
        if not self.structures(UnitTypeId.BANELINGNEST).ready:
            return False

        lings = self.units(UnitTypeId.ZERGLING).ready
        if not lings:
            return False

        issued = False
        for _ in range(max(1, n)):
            if not self.can_afford(UnitTypeId.BANELING):
                break
            lings = self.units(UnitTypeId.ZERGLING).ready
            if not lings:
                break
            zergling = lings.random
            self.do(zergling.train(UnitTypeId.BANELING))
            issued = True
        return issued

    async def train_overlord(self, n: int = 1) -> bool:
        if self.supply_cap >= 200:
            return False

        issued = False
        for _ in range(max(1, n)):
            if self.larva and self.can_afford(UnitTypeId.OVERLORD):
                if not self.larva:
                    break
                larva = self.larva.random
                self.do(larva.train(UnitTypeId.OVERLORD))
                issued = True
        return issued

    def _collect_metrics(self) -> dict:
        score = self.state.score
        # Track total minerals/gas gathered for rate calculation
        minerals_collected = float(getattr(score, "collected_minerals", getattr(score, "total_collected_minerals", 0.0)))
        gas_collected = float(getattr(score, "collected_vespene", getattr(score, "total_collected_vespene", 0.0)))
        # Enemy structure damage and destruction
        enemy_structures_destroyed = float(getattr(score, "killed_value_structures", 0.0))
        enemy_structure_damage = float(getattr(score, "total_damage_dealt_structures", 0.0))
        # High-value unit preservation (queens, lair, spire, etc.)
        preserved_units = float(self.units(UnitTypeId.QUEEN).amount)
        # Expansion: number of hatcheries built
        hatcheries = float(self.structures(UnitTypeId.HATCHERY).amount + self.structures(UnitTypeId.LAIR).amount + self.structures(UnitTypeId.HIVE).amount)
        return {
            "minerals": float(self.minerals),
            "gas": float(self.vespene),
            "workers": float(self.workers.amount),
            "army": float(
                self.units(UnitTypeId.ZERGLING).amount
                + self.units(UnitTypeId.BANELING).amount
            ),
            "killed_units": float(getattr(score, "killed_value_units", 0.0)),
            "killed_structures": enemy_structures_destroyed,
            "enemy_structure_damage": enemy_structure_damage,
            "preserved_units": preserved_units,
            "hatcheries": hatcheries,
            "lost_army": float(getattr(score, "lost_minerals_army", 0.0))
            + float(getattr(score, "lost_vespene_army", 0.0)),
            "lost_workers": float(getattr(score, "lost_minerals_economy", 0.0)),
            "minerals_collected": minerals_collected,
            "gas_collected": gas_collected,
            "game_time": float(self.time),
        }

    def _compute_step_reward(self, action_succeeded: bool) -> float:
        current = self._collect_metrics()
        if self._prev_metrics is None:
            self._prev_metrics = current
            return 0.0

        w = self._reward_weights
        delta = {
            k: current[k] - self._prev_metrics.get(k, 0)
            for k in ["minerals", "gas", "workers", "army"]
        }
        delta_kills = (current["killed_units"] + current["killed_structures"]) - (
            self._prev_metrics.get("killed_units", 0)
            + self._prev_metrics.get("killed_structures", 0)
        )
        delta_losses = current["lost_army"] - self._prev_metrics.get("lost_army", 0)
        delta_worker_loss = current["lost_workers"] - self._prev_metrics.get(
            "lost_workers", 0
        )

        # Compute resource acquisition rates (minerals/gas per second)
        curr_time = current["game_time"]
        curr_minerals_collected = current["minerals_collected"]
        curr_gas_collected = current["gas_collected"]
        mineral_rate = (curr_minerals_collected / curr_time) if curr_time > 0 else 0.0
        gas_rate = (curr_gas_collected / curr_time) if curr_time > 0 else 0.0

        # Enemy structure damage and destruction
        delta_enemy_structure_damage = current["enemy_structure_damage"] - self._prev_metrics.get("enemy_structure_damage", 0.0)
        delta_enemy_structures_destroyed = current["killed_structures"] - self._prev_metrics.get("killed_structures", 0.0)
        # Unit preservation (change in preserved units)
        delta_preserved_units = current["preserved_units"] - self._prev_metrics.get("preserved_units", 0.0)
        # Expansion (change in hatcheries)
        delta_hatcheries = current["hatcheries"] - self._prev_metrics.get("hatcheries", 0.0)

        reward = 0.0
        reward += w["minerals"] * delta["minerals"]
        reward += w["gas"] * delta["gas"]
        reward += w["workers"] * delta["workers"]
        reward += w["army"] * delta["army"]
        reward += w["kills"] * delta_kills
        reward += w["losses"] * delta_losses
        reward += w["worker_loss"] * delta_worker_loss
        # Add reward for resource acquisition rate
        reward += w["mineral_rate"] * mineral_rate
        reward += w["gas_rate"] * gas_rate
        # Add reward for enemy structure damage and destruction
        reward += w["enemy_structure_damage"] * delta_enemy_structure_damage
        reward += w["enemy_structure_destroyed"] * delta_enemy_structures_destroyed
        # Add reward for unit preservation (high-value units)
        reward += w["unit_preservation"] * delta_preserved_units
        # Add reward for expansion (new hatcheries)
        reward += w["expansion"] * delta_hatcheries

        if action_succeeded:
            reward += w["success"]
        if self.supply_left <= 0:
            reward += w["supply_penalty"]

        reward -= 0.0001  # survival bonus

        self._prev_metrics = current
        return float(max(-0.5, min(1.0, reward)))

    def _log_step(self, iteration: int, action: int, succeeded: bool, reward: float):
        if self._log_level < 1:
            return
        print(
            f"[Scaffold] iter={iteration} action={action} ok={succeeded} r={reward:.4f}"
        )

    def _log_episode_end(self, result, metrics: dict):
        if self._log_level < 1:
            return
        print(
            f"[Scaffold] episode ended result={result.name} "
            f"minerals={metrics['minerals']:.0f} gas={metrics['gas']:.0f} "
            f"workers={metrics['workers']:.0f} army={metrics['army']:.0f}"
        )

    def _log_train(self, total_loss: float, rl_loss: float, critic_loss: float):
        if self._log_level < 2:
            return
        print(
            f"[Scaffold] train loss: total={total_loss:.4f} rl={rl_loss:.4f} critic={critic_loss:.4f}"
        )

    def _log_entropy(self, entropy: float):
        if self._log_level < 2:
            return
        print(f"[Scaffold] entropy: {entropy:.4f}")

    def _get_under_saturated_extractor(self):
        ready_extractors = self.structures(UnitTypeId.EXTRACTOR).ready
        if not ready_extractors:
            return None

        candidates = []
        for extractor in ready_extractors:
            assigned = int(getattr(extractor, "assigned_harvesters", 0))
            ideal = int(getattr(extractor, "ideal_harvesters", 3))
            if assigned < max(1, ideal):
                candidates.append((assigned, extractor))

        if not candidates:
            return None

        candidates.sort(key=lambda pair: pair[0])
        return candidates[0][1]

    async def build_extractor(self) -> bool:
        if not self.can_afford(UnitTypeId.EXTRACTOR):
            return False

        for hatchery in self.townhalls.ready:
            for geyser in self.vespene_geyser.closer_than(10, hatchery):
                if self.structures(UnitTypeId.EXTRACTOR).closer_than(1.0, geyser):
                    continue
                worker = self.select_build_worker(geyser.position)
                if worker:
                    self.do(worker.build(UnitTypeId.EXTRACTOR, geyser))
                    return True
        return False
    
    async def build_spawning_pool(self) -> bool:
        if self.structures(UnitTypeId.SPAWNINGPOOL):
            return False
        if not self.structures(UnitTypeId.HATCHERY).ready:
            return False
        if not self.can_afford(UnitTypeId.SPAWNINGPOOL):
            return False
        hatch = self.structures(UnitTypeId.HATCHERY).ready.first
        pos = hatch.position.towards(self.game_info.map_center, 5)
        return await self.build(UnitTypeId.SPAWNINGPOOL, near=pos)
    
    async def build_roach_warren(self) -> bool:
        if self.structures(UnitTypeId.ROACHWARREN):
            return False
        if not self.structures(UnitTypeId.LAIR).ready:
            return False
        if not self.can_afford(UnitTypeId.ROACHWARREN):
            return False
        lair = self.structures(UnitTypeId.LAIR).ready.first
        pos = lair.position.towards(self.game_info.map_center, 5)
        return await self.build(UnitTypeId.ROACHWARREN, near=pos)
    
    async def attack_move(self, target: Point2 | None = None) -> bool:
        army = self.units.of_type({UnitTypeId.ZERGLING, UnitTypeId.BANELING})
        if not army:
            return False

        if self.enemy_units:
            attack_target: Point2 = self.enemy_units.closest_to(army.center).position
        elif self.enemy_structures:
            attack_target = self.enemy_structures.closest_to(army.center).position
        elif target is not None:
            attack_target = target
        else:
            attack_target = self.enemy_start_locations[0]

        for unit in army:
            if (
                not unit.can_attack_air
                and self.enemy_units.filter(lambda e: e.is_flying).exists
            ):
                ground_enemies = self.enemy_units.filter(lambda e: not e.is_flying)
                if ground_enemies.exists:
                    attack_target = ground_enemies.closest_to(army.center).position
            self.do(unit.attack(attack_target))

        return True
    
    async def build_baneling_nest(self) -> bool:
        if self.structures(UnitTypeId.BANELINGNEST):
            return False
        if not self.structures(UnitTypeId.LAIR).ready:
            return False
        if not self.can_afford(UnitTypeId.BANELINGNEST):
            return False
        lair = self.structures(UnitTypeId.LAIR).ready.first
        pos = lair.position.towards(self.game_info.map_center, 5)
        return await self.build(UnitTypeId.BANELINGNEST, near=pos)

    async def build_infestation_pit(self) -> bool:
        if self.structures(UnitTypeId.INFESTATIONPIT):
            return False
        if not self.structures(UnitTypeId.LAIR).ready:
            return False
        if not self.can_afford(UnitTypeId.INFESTATIONPIT):
            return False
        lair = self.structures(UnitTypeId.LAIR).ready.first
        pos = lair.position.towards(self.game_info.map_center, 5)
        return await self.build(UnitTypeId.INFESTATIONPIT, near=pos)

    async def build_greater_spire(self) -> bool:
        if self.structures(UnitTypeId.GREATERSPIRE):
            return False
        if not self.structures(UnitTypeId.SPIRE).ready:
            return False
        if not self.can_afford(UnitTypeId.GREATERSPIRE):
            return False
        spire = self.structures(UnitTypeId.SPIRE).ready.first
        pos = spire.position.towards(self.game_info.map_center, 5)
        return await self.build(UnitTypeId.GREATERSPIRE, near=pos)

    async def train_brood_lord(self, n: int = 1) -> bool:
        if not self.structures(UnitTypeId.GREATERSPIRE).ready:
            return False
        corruptors = self.units(UnitTypeId.CORRUPTOR).ready
        if not corruptors:
            return False
        issued = False
        for _ in range(max(1, n)):
            if not self.can_afford(UnitTypeId.BROODLORD):
                break
            corruptors = self.units(UnitTypeId.CORRUPTOR).ready
            if not corruptors:
                break
            corruptor = corruptors.random
            self.do(corruptor.train(UnitTypeId.BROODLORD))
            issued = True
        return issued

    async def build_spine_crawler(self) -> bool:
        if not self.can_afford(UnitTypeId.SPINECRAWLER):
            return False
        if not self.structures(UnitTypeId.HATCHERY).ready:
            return False
        hatch = self.structures(UnitTypeId.HATCHERY).ready.first
        pos = hatch.position.towards(self.game_info.map_center, 7)
        return await self.build(UnitTypeId.SPINECRAWLER, near=pos)

    async def build_spore_crawler(self) -> bool:
        if not self.can_afford(UnitTypeId.SPORECRAWLER):
            return False
        if not self.structures(UnitTypeId.HATCHERY).ready:
            return False
        hatch = self.structures(UnitTypeId.HATCHERY).ready.first
        pos = hatch.position.towards(self.game_info.map_center, 7)
        return await self.build(UnitTypeId.SPORECRAWLER, near=pos)

    async def inject_larva(self) -> bool:
        if self.units(UnitTypeId.QUEEN).ready and self.townhalls.ready:
            queen = self.units(UnitTypeId.QUEEN).ready.first
            hatch = self.townhalls.ready.first
            self.do(queen(AbilityId.EFFECT_INJECTLARVA, hatch))
            return True
        return False

    async def spread_creep(self) -> bool:
        if self.units(UnitTypeId.QUEEN).ready:
            queen = self.units(UnitTypeId.QUEEN).ready.first
            target_pos = queen.position.towards(self.game_info.map_center, 10)
            self.do(queen(AbilityId.BUILD_CREEPTUMOR_QUEEN, target_pos))
            return True
        return False

    async def transfuse(self) -> bool:
        if self.units(UnitTypeId.QUEEN).ready:
            queen = self.units(UnitTypeId.QUEEN).ready.first
            injured = self.units.filter(lambda u: u.health < u.health_max)
            if injured:
                self.do(queen(AbilityId.TRANSFUSION_TRANSFUSION, injured.first))
                return True
        return False

    async def research_zergling_speed(self) -> bool:
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready:
            pool = self.structures(UnitTypeId.SPAWNINGPOOL).ready.first
            self.do(pool.research(UpgradeId.ZERGLINGMOVEMENTSPEED))
            return True
        return False

    async def research_burrow(self) -> bool:
        if self.structures(UnitTypeId.HATCHERY).ready:
            hatch = self.structures(UnitTypeId.HATCHERY).ready.first
            self.do(hatch.research(UpgradeId.BURROW))
            return True
        return False

    async def research_roach_speed(self) -> bool:
        if self.structures(UnitTypeId.ROACHWARREN).ready:
            warren = self.structures(UnitTypeId.ROACHWARREN).ready.first
            self.do(warren.research(UpgradeId.GLIALRECONSTITUTION))
            return True
        return False

    async def research_baneling_speed(self) -> bool:
        if self.structures(UnitTypeId.BANELINGNEST).ready:
            nest = self.structures(UnitTypeId.BANELINGNEST).ready.first
            self.do(nest.research(UpgradeId.CENTRIFICALHOOKS))
            return True
        return False

    async def research_flyer_attacks(self) -> bool:
        if self.structures(UnitTypeId.SPIRE).ready:
            spire = self.structures(UnitTypeId.SPIRE).ready.first
            id = UpgradeId.ZERGFLYERWEAPONSLEVEL1
            self.do(spire.research(id))
            return True
        return False

    async def retreat(self) -> bool:
        # Only command idle combat units (self.army_units)
        for unit in self.units.idle.of_type(self.army_units):
            self.do(unit.move(self.start_location))
        return True

    async def regroup(self) -> bool:
        rally = self.start_location.towards(self.game_info.map_center, 8)
        for unit in self.units.idle.of_type(self.army_units):
            self.do(unit.move(rally))
        return True

    async def focus_fire(self) -> bool:
        if self.enemy_units and self.enemy_units.amount > 0:
            target = self.enemy_units.closest_to(self.start_location)
            for unit in self.units.idle.of_type(self.army_units):
                self.do(unit.attack(target))
            return True
        return False

    async def upgrade_to_spire(self) -> bool:
        """Upgrade Lair to Spire if possible."""
        # Must have Lair, can afford Spire, and no Spire exists
        if self.structures(UnitTypeId.SPIRE):
            return False
        if not self.structures(UnitTypeId.LAIR).ready:
            return False
        if not self.can_afford(UnitTypeId.SPIRE):
            return False
        lair = self.structures(UnitTypeId.LAIR).ready.first
        pos = lair.position.towards(self.game_info.map_center, 5)
        return await self.build(UnitTypeId.SPIRE, near=pos)

    async def upgrade_to_lair(self) -> bool:
        """Upgrade Hatchery to Lair if possible."""
        if self.structures(UnitTypeId.LAIR):
            return False
        hatcheries = self.structures(UnitTypeId.HATCHERY).ready
        if not hatcheries:
            return False
        if not self.can_afford(UnitTypeId.LAIR):
            return False
        hatch = hatcheries.first
        self.do(hatch.build(UnitTypeId.LAIR))
        return True

    async def morph_to_lair(self):
        """Morph a Hatchery to Lair if possible."""
        hatcheries = self.structures(UnitTypeId.HATCHERY).ready
        if not hatcheries:
            return False
        if self.structures(UnitTypeId.LAIR):
            return False
        if not self.can_afford(UnitTypeId.LAIR):
            return False
        hatch = hatcheries.first
        self.do(hatch.build(UnitTypeId.LAIR))
        return True

    async def morph_to_greater_spire(self):
        """Morph a Spire to Greater Spire if possible."""
        spires = self.structures(UnitTypeId.SPIRE).ready
        if not spires:
            return False
        if self.structures(UnitTypeId.GREATERSPIRE):
            return False
        if not self.can_afford(UnitTypeId.GREATERSPIRE):
            return False
        spire = spires.first
        self.do(spire.build(UnitTypeId.GREATERSPIRE))
        return True

    async def explore(self):
        scouts = self.units(UnitTypeId.ZERGLING).idle
        if not scouts:
            return False

        unexplored_expansions = [
            p for p in self.expansion_locations
            if self.state.visibility[int(p.x), int(p.y)] == 0
        ]

        if not unexplored_expansions:
            return False

        target = min(unexplored_expansions, key=lambda p: scouts.center.distance_to(p))
        if not scouts:
            return False
        self.do(scouts.random.move(target))
        return True
    
    async def send_drone_mineral(self):
        """Send an idle drone to mine minerals."""
        if not self.workers.idle:
            return False
        worker = self.workers.idle.random
        mineral_field = self.mineral_field.closest_to(worker)
        self.do(worker.gather(mineral_field))
        return True

    async def send_drone_gas(self):
        """Send an idle drone to mine gas."""
        if not self.workers.idle:
            return False
        extractors = self.structures(UnitTypeId.EXTRACTOR).ready
        if not extractors:
            return False
        extractor = extractors.random
        worker = self.workers.idle.random
        self.do(worker.gather(extractor))
        return True

    async def gather_vespene(self) -> bool:
        extractors = self.structures(UnitTypeId.EXTRACTOR).ready
        if not extractors or not self.workers:
            return False

        extractor = self._get_under_saturated_extractor()
        if extractor is None:
            return False

        nearby_workers = self.workers.closer_than(20, extractor)
        if nearby_workers and nearby_workers.idle:
            worker = nearby_workers.idle.random
        elif self.workers.idle:
            worker = self.workers.idle.random
        elif nearby_workers:
            worker = nearby_workers.random
        elif self.workers:
            worker = self.workers.random
        else:
            return False

        self.do(worker.gather(extractor))
        return True

    async def gather_minerals(self) -> bool:
        if not self.mineral_field or not self.workers:
            return False
        if self.workers.idle:
            worker = self.workers.idle.random
        elif self.workers:
            worker = self.workers.random
        else:
            return False
        mineral_field = self.mineral_field.closest_to(worker)
        self.do(worker.gather(mineral_field))
        return True

    async def maintain_worker_economy(self, max_idle_orders: int = 6):
        if max_idle_orders <= 0:
            return

        gatherable_workers = self.workers.filter(lambda w: not w.is_constructing_scv)

        if gatherable_workers.idle and self.mineral_field:
            for worker in gatherable_workers.idle[:max_idle_orders]:
                self.do(worker.gather(self.mineral_field.closest_to(worker)))

        if self.mineral_field:
            for extractor in self.structures(UnitTypeId.EXTRACTOR).ready:
                assigned = int(getattr(extractor, "assigned_harvesters", 0))
                ideal = int(getattr(extractor, "ideal_harvesters", 3))
                excess = max(0, assigned - ideal)
                if excess <= 0:
                    continue

                nearby_workers = gatherable_workers.closer_than(6, extractor)
                for worker in nearby_workers[:excess]:
                    self.do(worker.gather(self.mineral_field.closest_to(worker)))

    async def ensure_overlord_buffer(self, supply_buffer: int = 2) -> bool:
        if self.supply_left > max(1, int(supply_buffer)):
            return False
        if self.already_pending(UnitTypeId.OVERLORD):
            return False
        return await self.train_overlord(1)

    async def flood_zerglings(self, max_larva_spend: int = 6) -> int:
        if not self.structures(UnitTypeId.SPAWNINGPOOL).ready:
            return 0

        issued = 0
        for _ in range(max(1, int(max_larva_spend))):
            if not self.larva:
                break
            if self.supply_left < 2:
                break
            if not self.can_afford(UnitTypeId.ZERGLING):
                break
            larva = self.larva.random
            self.do(larva.train(UnitTypeId.ZERGLING))
            issued += 1
        return issued
    
    async def train_roach(self, n: int = 1) -> bool:
        if not self.structures(UnitTypeId.ROACHWARREN).ready:
            return False

        issued = False
        for _ in range(max(1, n)):
            if (
                self.larva
                and self.can_afford(UnitTypeId.ROACH)
                and self.supply_left >= 2
            ):
                if not self.larva:
                    break
                larva = self.larva.random
                self.do(larva.train(UnitTypeId.ROACH))
                issued = True
        return issued