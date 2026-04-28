from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
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
        self.observation_space = ((12, 64, 64), 9)
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
        # 42: train queen
        # 43: train ultralisk
        # 44: train infestor
        # 45: train lurker
        # 46: build evolution chamber
        # 47: build ultralisk cavern
        # 48: build nydus canal
        # 49: build lurker den
        # 50: research ground armor
        # 51: research air armor
        # 52: research neural parasite
        # 53: build overseer
        # 54: spawn changeling
        # 55: load nydus
        # 56: unload nydus
        self.total_actions = 57

        # Rebalanced rewards: scaled up to be meaningful relative to terminal
        # reward of ±50. Shaped rewards should sum to ~50 over a full game.
        self._reward_weights = reward_weights or {
            "workers": 0.05,                       # per-worker incremental reward (5x)
            "army": 0.10,                          # per-army-unit incremental reward (5x)
            "losses": -0.15,                       # penalize losses (5x)
            "worker_loss": -0.10,
            "success": 0.05,                       # bonus for successful actions (5x)
            "supply_penalty": -0.10,
            "enemy_unit_kills": 0.50,              # reward kills more strongly (5x)
            "enemy_structures_destroyed": 2.50,    # structures matter more (5x)
            "expansion": 0.50,
            "structure_built": 0.10,
            "low_worker_penalty": -0.05,
            "worker_saturation": 0.025,
            "workers_per_hatchery": 0.25,
            "worker_milestone_32": 0.50,
            "worker_milestone_60": 1.00,
            "attack_action": 0.25,
            "enemy_base_proximity": 0.25,
            "army_movement": 0.05,
            "queen_exists": 0.15,
        }
        self._log_level = log_level
        self.army_units = [UnitTypeId.ZERGLING, UnitTypeId.BANELING, UnitTypeId.HYDRALISK, UnitTypeId.MUTALISK, UnitTypeId.ROACH, UnitTypeId.CORRUPTOR, UnitTypeId.BROODLORD, UnitTypeId.ULTRALISK, UnitTypeId.INFESTOR, UnitTypeId.LURKERMP]
        self.worker_types = [UnitTypeId.DRONE, UnitTypeId.PROBE, UnitTypeId.SCV]
        self.structure_types = [UnitTypeId.SPAWNINGPOOL, UnitTypeId.HYDRALISKDEN, UnitTypeId.SPIRE, UnitTypeId.ROACHWARREN, UnitTypeId.BANELINGNEST, UnitTypeId.INFESTATIONPIT, UnitTypeId.GREATERSPIRE, UnitTypeId.SPINECRAWLER, UnitTypeId.SPORECRAWLER, UnitTypeId.EVOLUTIONCHAMBER, UnitTypeId.ULTRALISKCAVERN, UnitTypeId.NYDUSCANAL, UnitTypeId.LURKERDENMP, UnitTypeId.HATCHERY, UnitTypeId.LAIR, UnitTypeId.HIVE]
        self._cumulative_stats = {
            "workers_created": 0,
            "army_units_created": 0,
            "structures_built": 0,
            "units_lost": 0,
            "structures_lost": 0,
            "enemy_units_killed": 0,
            "enemy_structures_destroyed": 0,
            "enemy_units_killed_value": 0.0,
            "enemy_structures_destroyed_value": 0.0,
            "units_lost_value": 0.0,
            "structures_lost_value": 0.0,
        }
        self._prev_metrics = None
        self._prev_minerals = None
        self._prev_time = None
        self._prev_score_cumulative = None

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
        worker_count = self.units(UnitTypeId.DRONE).amount / 50.0
        army_count = (
            max(
                self.units.not_flying.amount - self.units(UnitTypeId.DRONE).amount,
                0,
            )
            / 100.0
        )
        drone_supply = (self.units(UnitTypeId.DRONE).amount * 1) / 200.0
        income_rate = 0.0
        if self._prev_minerals is not None and self._prev_time is not None and self.time > self._prev_time:
            dt = self.time - self._prev_time
            income_rate = min((self.minerals - self._prev_minerals) / max(dt, 0.1), 1000) / 1000.0
        resources = np.array(
            [
                gas_available,
                minerals_available,
                supply_used,
                supply_cap,
                game_time,
                worker_count,
                army_count,
                drone_supply,
                income_rate,
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
            return await self.train_zerglings(3)
        if action_idx == 3:
            return await self.attack_move()
        if action_idx == 4:
            return await self.train_drones(2)
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
        if action_idx == 42:
            return await self.train_queen(1)
        if action_idx == 43:
            return await self.train_ultralisk(1)
        if action_idx == 44:
            return await self.train_infestor(1)
        if action_idx == 45:
            return await self.train_lurker(1)
        if action_idx == 46:
            return await self.build_evolution_chamber()
        if action_idx == 47:
            return await self.build_ultralisk_cavern()
        if action_idx == 48:
            return await self.build_nydus_canal()
        if action_idx == 49:
            return await self.build_lurker_den()
        if action_idx == 50:
            return await self.research_ground_armor()
        if action_idx == 51:
            return await self.research_air_armor()
        if action_idx == 52:
            return await self.research_neural_parasite()
        if action_idx == 53:
            return await self.build_overseer()
        if action_idx == 54:
            return await self.spawn_changeling()
        if action_idx == 55:
            return await self.load_nydus()
        if action_idx == 56:
            return await self.unload_nydus()
        return False

    def get_action_mask(self) -> torch.Tensor:
        """Return a boolean mask where True = action is legal, False = illegal."""
        mask = torch.ones(self.total_actions, dtype=torch.bool)
        
        has_pool = bool(self.structures(UnitTypeId.SPAWNINGPOOL).ready)
        has_lair = bool(self.structures(UnitTypeId.LAIR).ready)
        has_hive = bool(self.structures(UnitTypeId.HIVE).ready)
        has_hydra_den = bool(self.structures(UnitTypeId.HYDRALISKDEN).ready)
        has_spire = bool(self.structures(UnitTypeId.SPIRE).ready)
        has_roach_warren = bool(self.structures(UnitTypeId.ROACHWARREN).ready)
        has_baneling_nest = bool(self.structures(UnitTypeId.BANELINGNEST).ready)
        has_infestation_pit = bool(self.structures(UnitTypeId.INFESTATIONPIT).ready)
        has_greater_spire = bool(self.structures(UnitTypeId.GREATERSPIRE).ready)
        has_ultralisk_cavern = bool(self.structures(UnitTypeId.ULTRALISKCAVERN).ready)
        has_lurker_den = bool(self.structures(UnitTypeId.LURKERDENMP).ready)
        has_nydus = bool(self.structures(UnitTypeId.NYDUSCANAL).ready)
        has_overseer = bool(self.units(UnitTypeId.OVERSEER).ready)
        has_overlord = bool(self.units(UnitTypeId.OVERLORD).ready)
        has_larva = bool(self.larva)
        has_hatchery = bool(self.townhalls.ready)
        has_extractor = bool(self.structures(UnitTypeId.EXTRACTOR).ready)
        has_workers = bool(self.workers)
        has_army = bool(self.units.of_type(self.army_units).ready)
        has_hydras = bool(self.units(UnitTypeId.HYDRALISK).ready)
        has_queen = bool(self.units(UnitTypeId.QUEEN).ready)
        has_zergling = bool(self.units(UnitTypeId.ZERGLING).ready)
        
        can_afford_pool = self.can_afford(UnitTypeId.SPAWNINGPOOL)
        can_afford_lair = self.can_afford(UnitTypeId.LAIR)
        
        # 0: no-op — always valid
        
        # 1: build spawning pool
        if has_pool or not has_hatchery or not can_afford_pool:
            mask[1] = False
        
        # 2: train zerglings
        if not has_pool or not has_larva:
            mask[2] = False
        
        # 3: attack
        if not has_army:
            mask[3] = False
        
        # 4: train drones
        if not has_larva:
            mask[4] = False
        
        # 5: train overlord
        if not has_larva:
            mask[5] = False
        
        # 6: train hydralisk (anti-air)
        if not has_hydra_den or not has_larva:
            mask[6] = False
        
        # 7: train mutalisk (flying)
        if not has_spire or not has_larva:
            mask[7] = False
        
        # 8: build hydralisk den
        if has_hydra_den or not has_lair:
            mask[8] = False
        
        # 9: build spire
        if has_spire or not has_lair:
            mask[9] = False
        
        # 10: build roach warren
        if has_roach_warren or not has_pool:
            mask[10] = False
        
        # 11: train roach
        if not has_roach_warren or not has_larva:
            mask[11] = False
        
        # 12: build baneling nest
        if has_baneling_nest or not has_pool:
            mask[12] = False
        
        # 13: train baneling
        if not has_baneling_nest or not has_larva or not has_zergling:
            mask[13] = False
        
        # 14: build infestation pit
        if has_infestation_pit or not has_lair:
            mask[14] = False
        
        # 15: build greater spire
        if has_greater_spire or not has_spire or not has_hive:
            mask[15] = False
        
        # 16: train brood lord
        if not has_greater_spire or not has_larva:
            mask[16] = False
        
        # 17: build spine crawler
        if not has_hatchery or not has_workers:
            mask[17] = False
        
        # 18: build spore crawler
        if not has_hatchery or not has_workers:
            mask[18] = False
        
        # 19: inject larva
        if not has_queen:
            mask[19] = False
        
        # 20: spread creep
        if not has_queen:
            mask[20] = False
        
        # 21: transfuse
        if not has_queen:
            mask[21] = False
        
        # 22: research zergling speed
        if not has_pool:
            mask[22] = False
        
        # 23: research burrow
        if not has_lair:
            mask[23] = False
        
        # 24: research roach speed
        if not has_roach_warren:
            mask[24] = False
        
        # 25: research baneling speed
        if not has_baneling_nest:
            mask[25] = False
        
        # 26: research flyer attacks
        if not has_spire:
            mask[26] = False
        
        # 27: retreat
        if not has_army:
            mask[27] = False
        
        # 28: regroup
        if not has_army:
            mask[28] = False
        
        # 29: focus fire
        if not has_army:
            mask[29] = False
        
        # 30: morph to lair
        if has_lair or has_hive or not has_pool or not can_afford_lair:
            mask[30] = False
        
        # 31: morph to greater spire
        if has_greater_spire or not has_spire or not has_hive:
            mask[31] = False
        
        # 32: send drone to minerals
        if not has_workers or not self.mineral_field:
            mask[32] = False
        
        # 33: send drone to gas
        if not has_workers or not has_extractor:
            mask[33] = False
        
        # 34: explore with zergling
        if not has_zergling:
            mask[34] = False
        
        # 35: build extractor
        if has_extractor or not has_hatchery or not has_workers:
            mask[35] = False
        
        # 36: expand
        if not self.can_afford(UnitTypeId.HATCHERY):
            mask[36] = False
        
        # 37: gather vespene
        if not has_extractor or not has_workers:
            mask[37] = False
        
        # 38: gather minerals
        if not has_workers or not self.mineral_field:
            mask[38] = False
        
        # 39: maintain worker economy — always valid
        
        # 40: ensure overlord buffer — always valid
        
        # 41: flood zerglings
        if not has_pool or not has_larva:
            mask[41] = False
        
        # 42: train queen
        if not has_pool or not has_hatchery:
            mask[42] = False
        
        # 43: train ultralisk
        if not has_ultralisk_cavern or not has_larva:
            mask[43] = False
        
        # 44: train infestor
        if not has_infestation_pit or not has_larva:
            mask[44] = False
        
        # 45: train lurker
        if not has_lurker_den or not has_hydras:
            mask[45] = False
        
        # 46: build evolution chamber
        if self.structures(UnitTypeId.EVOLUTIONCHAMBER).exists or not has_pool:
            mask[46] = False
        
        # 47: build ultralisk cavern
        if has_ultralisk_cavern or not has_hive:
            mask[47] = False
        
        # 48: build nydus canal
        if has_nydus or not has_hive:
            mask[48] = False
        
        # 49: build lurker den
        if has_lurker_den or not has_lair:
            mask[49] = False
        
        # 50: research ground armor
        if not self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready:
            mask[50] = False
        
        # 51: research air armor
        if not self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready:
            mask[51] = False
        
        # 52: research neural parasite
        if not has_infestation_pit:
            mask[52] = False
        
        # 53: build overseer
        if has_overseer or not has_overlord or (not has_lair and not has_hive):
            mask[53] = False
        
        # 54: spawn changeling
        if not has_overseer:
            mask[54] = False
        
        # 55: load nydus
        if not has_nydus or not has_army:
            mask[55] = False
        
        # 56: unload nydus
        if not has_nydus:
            mask[56] = False
        
        return mask

    async def build_hydralisk_den(self):
        if self.structures(UnitTypeId.HYDRALISKDEN).exists:
            return False
        if self.can_afford(UnitTypeId.HYDRALISKDEN) and self.structures(UnitTypeId.LAIR).ready:
            lair = self.structures(UnitTypeId.LAIR).ready.first
            pos = lair.position.towards(self.game_info.map_center, 5)
            await self.build(UnitTypeId.HYDRALISKDEN, near=pos)
            return True
        return False

    async def build_spire(self):
        if self.structures(UnitTypeId.SPIRE).exists:
            return False
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
        if self.can_afford(UnitTypeId.MUTALISK) and self.structures(UnitTypeId.SPIRE).ready:
            for _ in range(n):
                available_larva = self.units(UnitTypeId.LARVA).filter(lambda l: l.tag not in self.unit_tags_received_action)
                if not available_larva:
                    return False
                larva = available_larva.first
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

    async def train_queen(self, n: int = 1) -> bool:
        if not self.structures(UnitTypeId.SPAWNINGPOOL).ready:
            return False

        issued = False
        for _ in range(max(1, n)):
            if not self.can_afford(UnitTypeId.QUEEN):
                break
            if self.already_pending(UnitTypeId.QUEEN) >= self.townhalls.amount:
                break
            for hatchery in self.townhalls.ready:
                self.do(hatchery.train(UnitTypeId.QUEEN))
                issued = True
                break
        return issued

    async def train_ultralisk(self, n: int = 1) -> bool:
        if not self.structures(UnitTypeId.ULTRALISKCAVERN).ready:
            return False

        issued = False
        for _ in range(max(1, n)):
            if (
                self.larva
                and self.can_afford(UnitTypeId.ULTRALISK)
                and self.supply_left >= 6
            ):
                if not self.larva:
                    break
                larva = self.larva.random
                self.do(larva.train(UnitTypeId.ULTRALISK))
                issued = True
        return issued

    async def train_infestor(self, n: int = 1) -> bool:
        if not self.structures(UnitTypeId.INFESTATIONPIT).ready:
            return False

        issued = False
        for _ in range(max(1, n)):
            if (
                self.larva
                and self.can_afford(UnitTypeId.INFESTOR)
                and self.supply_left >= 2
            ):
                if not self.larva:
                    break
                larva = self.larva.random
                self.do(larva.train(UnitTypeId.INFESTOR))
                issued = True
        return issued

    async def train_lurker(self, n: int = 1) -> bool:
        if not self.structures(UnitTypeId.LURKERDENMP).ready:
            return False

        hydralisks = self.units(UnitTypeId.HYDRALISK).ready
        if not hydralisks:
            return False

        issued = False
        for _ in range(max(1, n)):
            if not self.can_afford(UnitTypeId.LURKERMP):
                break
            hydralisks = self.units(UnitTypeId.HYDRALISK).ready
            if not hydralisks:
                break
            hydra = hydralisks.random
            self.do(hydra.train(UnitTypeId.LURKERMP))
            issued = True
        return issued

    async def build_overseer(self) -> bool:
        if not self.can_afford(UnitTypeId.OVERSEER):
            return False
        if not self.structures(UnitTypeId.LAIR).ready and not self.structures(UnitTypeId.HIVE).ready:
            return False
        overlords = self.units(UnitTypeId.OVERLORD).ready.idle
        if not overlords:
            return False
        overlord = overlords.first
        self.do(overlord(AbilityId.MORPH_OVERSEER))
        return True

    async def spawn_changeling(self) -> bool:
        overseers = self.units(UnitTypeId.OVERSEER).ready
        if not overseers:
            return False
        overseer = overseers.first
        if overseer.energy >= 50:
            # Ability enum name differs across python-sc2 versions.
            ability = getattr(AbilityId, "SPAWNCHANGELING_SPAWNCHANGELING", None)
            if ability is None:
                ability = getattr(AbilityId, "SPAWNCHANGELING_CHANGELING", None)
            if ability is None:
                return False
            self.do(overseer(ability))
            return True
        return False

    async def load_nydus(self) -> bool:
        nyduses = self.structures(UnitTypeId.NYDUSCANAL).ready
        if not nyduses:
            return False
        army = self.units.of_type(self.army_units).ready.idle
        if not army:
            return False
        loaded = 0
        for nydus in nyduses:
            for unit in army:
                if unit.distance_to(nydus) <= 15:
                    self.do(unit.load(nydus))
                    loaded += 1
        return loaded > 0

    async def unload_nydus(self) -> bool:
        nyduses = self.structures(UnitTypeId.NYDUSCANAL).ready
        if not nyduses:
            return False
        unloaded = 0
        for nydus in nyduses:
            if nydus.cargo_used > 0:
                self.do(nydus.unload_all())
                unloaded += 1
        return unloaded > 0

    def _collect_metrics(self) -> dict:
        score = self.state.score
        
        army_count = sum(self.units(u).amount for u in self.army_units)
        
        enemy_kills = float(getattr(score, "killed_minerals_units", 0.0)) + float(getattr(score, "killed_vespene_units", 0.0))
        structure_kills = float(getattr(score, "killed_minerals_structures", 0.0)) + float(getattr(score, "killed_vespene_structures", 0.0))
        
        lost_army = float(getattr(score, "lost_minerals_army", 0.0)) + float(getattr(score, "lost_vespene_army", 0.0))
        lost_workers = float(getattr(score, "lost_minerals_economy", 0.0)) + float(getattr(score, "lost_vespene_economy", 0.0))
        
        # Use race-agnostic townhall count to avoid bias for non-Zerg bots.
        hatcheries = float(self.townhalls.amount)
        
        supply_used = self.supply_used
        supply_cap = self.supply_cap
        
        return {
            "workers": float(self.workers.amount),
            "army": float(army_count),
            "minerals": float(self.minerals),
            "gas": float(self.vespene),
            "enemy_unit_kills": enemy_kills,
            "enemy_structures_destroyed": structure_kills,
            "lost_army": lost_army,
            "lost_workers": lost_workers,
            "hatcheries": hatcheries,
            "supply_used": float(supply_used),
            "supply_cap": float(supply_cap),
            "game_time": float(self.time),
            "structures": float(self.structures.amount),
        }

    def _compute_step_reward(self, action_succeeded: bool, action_idx: int = -1) -> float:
        current = self._collect_metrics()
        self._sync_cumulative_score_fallback()
        if self._prev_metrics is None:
            self._prev_metrics = current
            self._prev_minerals = self.minerals
            self._prev_time = self.time
            return 0.0

        w = self._reward_weights

        delta_workers = current["workers"] - self._prev_metrics.get("workers", 0)
        delta_army = current["army"] - self._prev_metrics.get("army", 0)
        delta_enemy_unit_kills = current["enemy_unit_kills"] - self._prev_metrics.get("enemy_unit_kills", 0)
        delta_enemy_structures_destroyed = current.get("enemy_structures_destroyed", 0) - self._prev_metrics.get("enemy_structures_destroyed", 0)
        delta_losses = current["lost_army"] - self._prev_metrics.get("lost_army", 0)
        delta_worker_loss = current["lost_workers"] - self._prev_metrics.get("lost_workers", 0)
        delta_hatcheries = current["hatcheries"] - self._prev_metrics.get("hatcheries", 0)
        delta_structures = current["structures"] - self._prev_metrics.get("structures", 0)

        # NOTE: cumulative stats are maintained by event handlers
        # (on_unit_created, on_building_construction_complete, on_unit_destroyed).
        # Avoid updating those counters here to prevent double-counting.

        reward = 0.0
        reward += w.get("workers", 0) * delta_workers
        reward += w.get("army", 0) * delta_army
        reward += w.get("losses", 0) * delta_losses
        reward += w.get("worker_loss", 0) * delta_worker_loss
        reward += w.get("enemy_unit_kills", 0) * delta_enemy_unit_kills
        reward += w.get("enemy_structures_destroyed", 0) * delta_enemy_structures_destroyed
        reward += w.get("expansion", 0) * delta_hatcheries

        if action_succeeded:
            reward += w.get("success", 0)

        if self.units(UnitTypeId.QUEEN).ready:
            reward += w.get("queen_exists", 0)

        # Penalize deviation from ideal worker saturation, with a deadzone
        # to avoid constant negative baseline when near target.
        ideal_workers = current.get("hatcheries", 0) * 16
        saturation_gap = abs(current.get("workers", 0) - ideal_workers)
        if ideal_workers > 0:
            saturation_penalty = max(0.0, saturation_gap - 2.0) / float(max(1.0, ideal_workers))
        else:
            saturation_penalty = 0.0
        reward -= w.get("worker_saturation", 0) * saturation_penalty

        target_workers_per_base = 16
        if current["hatcheries"] > 0:
            workers_per_base = current["workers"] / current["hatcheries"]
            if workers_per_base >= target_workers_per_base:
                reward += w["workers_per_hatchery"]

        if action_succeeded and action_idx == 3:
            reward += w["attack_action"]

        if delta_structures > 0:
            reward += w["structure_built"]

        if current["workers"] < 5 and current["hatcheries"] >= 1:
            reward += w["low_worker_penalty"]

        if self.supply_left <= 0 and self.supply_cap > 0:
            reward += w["supply_penalty"]

        enemy_base = self.enemy_start_locations[0] if self.enemy_start_locations else None
        if enemy_base:
            army = self.units.of_type(self.army_units)
            if army:
                army_center = army.center
                dist_to_enemy = army_center.distance_to(enemy_base)
                if dist_to_enemy < 20:
                    reward += w["enemy_base_proximity"] * (1.0 - dist_to_enemy / 20)

        current_army = self.units.of_type(self.army_units).ready
        if current_army:
            idle_army = current_army.idle
            if idle_army and action_idx in (27, 28):
                reward += w["army_movement"]

        wn = current["workers"]
        # Reduced milestone set: keep only larger, meaningful milestones
        if wn >= 32 and not getattr(self, '_milestone_32_achieved', False):
            self._milestone_32_achieved = True
            reward += w.get("worker_milestone_32", 0)
        if wn >= 60 and not getattr(self, '_milestone_60_achieved', False):
            self._milestone_60_achieved = True
            reward += w.get("worker_milestone_60", 0)

        self._prev_metrics = current
        self._prev_minerals = self.minerals
        self._prev_time = self.time
        # No clipping: return raw shaped reward to preserve full signal
        return float(reward)

    def _sync_cumulative_score_fallback(self):
        """Update value-based cumulative stats from score deltas.

        This is a resilient fallback for matches where unit-destroyed callbacks
        may miss some deaths (e.g. fog-of-war or API visibility timing).
        """
        score = self.state.score
        current_score_totals = {
            "enemy_units_killed_value": float(getattr(score, "killed_minerals_units", 0.0))
            + float(getattr(score, "killed_vespene_units", 0.0)),
            "enemy_structures_destroyed_value": float(getattr(score, "killed_minerals_structures", 0.0))
            + float(getattr(score, "killed_vespene_structures", 0.0)),
            "units_lost_value": float(getattr(score, "lost_minerals_army", 0.0))
            + float(getattr(score, "lost_vespene_army", 0.0))
            + float(getattr(score, "lost_minerals_economy", 0.0))
            + float(getattr(score, "lost_vespene_economy", 0.0)),
            "structures_lost_value": float(getattr(score, "lost_minerals_technology", 0.0))
            + float(getattr(score, "lost_vespene_technology", 0.0)),
        }

        if self._prev_score_cumulative is None:
            self._prev_score_cumulative = current_score_totals
            return

        for key, total in current_score_totals.items():
            prev_total = float(self._prev_score_cumulative.get(key, 0.0))
            delta = max(0.0, total - prev_total)
            self._cumulative_stats[key] = float(self._cumulative_stats.get(key, 0.0)) + delta

        self._prev_score_cumulative = current_score_totals

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
        if not self.townhalls.ready:
            return False
        if not self.can_afford(UnitTypeId.SPAWNINGPOOL):
            return False
        townhall = self.townhalls.ready.first
        pos = townhall.position.towards(self.game_info.map_center, 5)
        return await self.build(UnitTypeId.SPAWNINGPOOL, near=pos)

    async def build_evolution_chamber(self) -> bool:
        if self.structures(UnitTypeId.EVOLUTIONCHAMBER):
            return False
        if not self.townhalls.ready:
            return False
        if not self.can_afford(UnitTypeId.EVOLUTIONCHAMBER):
            return False
        townhall = self.townhalls.ready.first
        pos = townhall.position.towards(self.game_info.map_center, 5)
        return await self.build(UnitTypeId.EVOLUTIONCHAMBER, near=pos)

    async def build_ultralisk_cavern(self) -> bool:
        if self.structures(UnitTypeId.ULTRALISKCAVERN):
            return False
        if not self.structures(UnitTypeId.HIVE).ready:
            return False
        if not self.can_afford(UnitTypeId.ULTRALISKCAVERN):
            return False
        hive = self.structures(UnitTypeId.HIVE).ready.first
        pos = hive.position.towards(self.game_info.map_center, 5)
        return await self.build(UnitTypeId.ULTRALISKCAVERN, near=pos)

    async def build_nydus_canal(self) -> bool:
        if not self.can_afford(UnitTypeId.NYDUSCANAL):
            return False
        if not self.townhalls.ready:
            return False
        townhall = self.townhalls.ready.first
        pos = townhall.position.towards(self.game_info.map_center, 5)
        return await self.build(UnitTypeId.NYDUSCANAL, near=pos)

    async def build_lurker_den(self) -> bool:
        if self.structures(UnitTypeId.LURKERDENMP).exists:
            return False
        if not self.structures(UnitTypeId.LAIR).ready:
            return False
        if not self.can_afford(UnitTypeId.LURKERDENMP):
            return False
        lair = self.structures(UnitTypeId.LAIR).ready.first
        pos = lair.position.towards(self.game_info.map_center, 5)
        return await self.build(UnitTypeId.LURKERDENMP, near=pos)

    async def build_roach_warren(self) -> bool:
        if self.structures(UnitTypeId.ROACHWARREN).exists:
            return False
        if not self.structures(UnitTypeId.SPAWNINGPOOL).ready:
            return False
        if not self.can_afford(UnitTypeId.ROACHWARREN):
            return False
        if not self.townhalls.ready:
            return False
        townhall = self.townhalls.ready.first
        pos = townhall.position.towards(self.game_info.map_center, 5)
        return await self.build(UnitTypeId.ROACHWARREN, near=pos)
    
    async def attack_move(self, target: Point2 | None = None) -> bool:
        army = self.units.of_type(set(self.army_units)).ready
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
        if not self.structures(UnitTypeId.SPAWNINGPOOL).ready:
            return False
        if not self.can_afford(UnitTypeId.BANELINGNEST):
            return False
        if not self.townhalls.ready:
            return False
        townhall = self.townhalls.ready.first
        pos = townhall.position.towards(self.game_info.map_center, 5)
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
        # Avoid repeatedly building spines: skip if one exists or is pending
        if self.structures(UnitTypeId.SPINECRAWLER).exists or self.already_pending(UnitTypeId.SPINECRAWLER):
            return False
        if not self.can_afford(UnitTypeId.SPINECRAWLER):
            return False
        if not self.townhalls.ready:
            return False
        townhall = self.townhalls.ready.first
        pos = townhall.position.towards(self.game_info.map_center, 7)
        return await self.build(UnitTypeId.SPINECRAWLER, near=pos)

    async def build_spore_crawler(self) -> bool:
        # Avoid repeatedly building spores: skip if one exists or is pending
        if self.structures(UnitTypeId.SPORECRAWLER).exists or self.already_pending(UnitTypeId.SPORECRAWLER):
            return False
        if not self.can_afford(UnitTypeId.SPORECRAWLER):
            return False
        if not self.townhalls.ready:
            return False
        townhall = self.townhalls.ready.first
        pos = townhall.position.towards(self.game_info.map_center, 7)
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
        if self.townhalls.ready:
            townhall = self.townhalls.ready.first
            self.do(townhall.research(UpgradeId.BURROW))
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

    async def research_ground_armor(self) -> bool:
        if not self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready:
            return False
        if not self.can_afford(UnitTypeId.EVOLUTIONCHAMBER):
            return False
        evo = self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready.first
        self.do(evo.research(UpgradeId.ZERGGROUNDARMORSLEVEL1))
        return True

    async def research_air_armor(self) -> bool:
        if not self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready:
            return False
        if not self.can_afford(UnitTypeId.EVOLUTIONCHAMBER):
            return False
        evo = self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready.first
        self.do(evo.research(UpgradeId.ZERGFLYERARMORSLEVEL1))
        return True

    async def research_neural_parasite(self) -> bool:
        if self.structures(UnitTypeId.INFESTATIONPIT).ready:
            pit = self.structures(UnitTypeId.INFESTATIONPIT).ready.first
            self.do(pit.research(UpgradeId.NEURALPARASITE))
            return True
        return False

    async def retreat(self) -> bool:
        if self.units.idle.of_type(self.army_units).amount == 0:
            return False
        if self.start_location is None:
            return False
        for unit in self.units.idle.of_type(self.army_units):
            self.do(unit.move(self.start_location))
        return True

    async def regroup(self) -> bool:
        if self.start_location is None:
            return False
        rally = self.start_location.towards(self.game_info.map_center, 8)
        for unit in self.units.idle.of_type(self.army_units):
            self.do(unit.move(rally))
        return True

    async def focus_fire(self) -> bool:
        if self.enemy_units and self.enemy_units.amount > 0:
            if self.start_location is None:
                return False
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
        if self.start_location is None:
            return False
        hatch = hatcheries.closest_to(self.start_location)
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

        ideal_workers = self.structures(UnitTypeId.HATCHERY).amount * 16
        if self.units(UnitTypeId.LAIR).ready:
            ideal_workers += 16
        if self.units(UnitTypeId.HIVE).ready:
            ideal_workers += 16

        if self.workers.amount < ideal_workers and self.larva and self.can_afford(UnitTypeId.DRONE) and self.supply_left > 0:
            larva = self.larva.random
            self.do(larva.train(UnitTypeId.DRONE))

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
    
    def get_performance_metrics(self) -> dict:  
        self._sync_cumulative_score_fallback()
        score = self.state.score  
        
        economic_metrics = {
            "mineral_collection_efficiency": score.collected_minerals / max(score.spent_minerals, 1),
            "vespene_collection_efficiency": score.collected_vespene / max(score.spent_vespene, 1),  
            "idle_worker_time": score.idle_worker_time,  
            "idle_production_time": score.idle_production_time,  
        }  
        
        military_metrics = {  
            "total_damage_dealt": score.total_damage_dealt_life + score.total_damage_dealt_shields,  
            "total_damage_taken": score.total_damage_taken_life + score.total_damage_taken_shields,  
            "damage_ratio": (score.total_damage_dealt_life + score.total_damage_dealt_shields) /   
                        max(score.total_damage_taken_life + score.total_damage_taken_shields, 1),  
            "kill_value_ratio": score.killed_value_units / max(  
                score.lost_minerals_army + score.lost_vespene_army +   
                score.lost_minerals_economy + score.lost_vespene_economy, 1),
        }  
        
        resource_metrics = {  
            "total_resources_collected": score.collected_minerals + score.collected_vespene,
            "total_resources_spent": score.spent_minerals + score.spent_vespene,  
            "resource_spending_rate": (score.spent_minerals + score.spent_vespene) /   
                                    max(score.collected_minerals + score.collected_vespene, 1),
        }  
        
        production_metrics = {  
            "total_unit_value": score.total_value_units,  
            "total_structure_value": score.total_value_structures,  
            "total_value_created": score.total_value_units + score.total_value_structures,  
            "value_lost_units": (score.lost_minerals_army + score.lost_vespene_army +   
                            score.lost_minerals_economy + score.lost_vespene_economy),  
            "value_lost_structures": score.lost_minerals_technology + score.lost_vespene_technology,  
            "net_value_retained": (score.total_value_units + score.total_value_structures) -   
                                ((score.lost_minerals_army + score.lost_vespene_army +   
                                score.lost_minerals_economy + score.lost_vespene_economy) +  
                                (score.lost_minerals_technology + score.lost_vespene_technology)),  
        }  
        
        army_composition = {  
            "workers": self.workers.amount,  
            "army_count": self.units.amount - self.workers.amount,  
            "structure_count": self.structures.amount,  
            "supply_army": self.supply_army,  
            "supply_workers": self.supply_workers,  
            "supply_economy": score.food_used_economy,  
            "supply_technology": score.food_used_technology,  
        }  
        
        return {  
            "economic": economic_metrics,  
            "military": military_metrics,   
            "resources": resource_metrics,  
            "production": production_metrics,  
            "composition": army_composition,  
            "game_time": self.time,  
            "cumulative": self._cumulative_stats.copy(),
        }

    async def on_unit_created(self, unit: Unit):
        await super().on_unit_created(unit)
        unit_type = unit.type_id
        
        if unit_type in self.worker_types:
            self._cumulative_stats["workers_created"] += 1
        elif unit_type in self.army_units:
            self._cumulative_stats["army_units_created"] += 1
        # Structure build completions are tracked in on_building_construction_complete

    async def on_building_construction_complete(self, unit: Unit):
        await super().on_building_construction_complete(unit)
        if unit.type_id in self.structure_types:
            self._cumulative_stats["structures_built"] += 1

    async def on_unit_destroyed(self, unit_tag: int):  
        await super().on_unit_destroyed(unit_tag)  
        
        unit_map = getattr(self, "_all_units_previous_map", None)
        if unit_map is None:
            return
        unit = unit_map.get(unit_tag)  
        if not unit:  
            return
        
        if unit.is_enemy:  
            if unit.type_id in self.army_units or unit.type_id in self.worker_types:  
                self._cumulative_stats["enemy_units_killed"] += 1  
            elif unit.type_id in self.structure_types:  
                self._cumulative_stats["enemy_structures_destroyed"] += 1  
        elif unit.is_mine:  
            if unit.type_id in self.army_units or unit.type_id in self.worker_types:  
                self._cumulative_stats["units_lost"] += 1  
            elif unit.type_id in self.structure_types:  
                self._cumulative_stats["structures_lost"] += 1

    def reset_cumulative_stats(self):
        self._cumulative_stats = {
            "workers_created": 0,
            "army_units_created": 0,
            "structures_built": 0,
            "enemy_units_killed": 0,
            "enemy_structures_destroyed": 0,
            "units_lost": 0,
            "structures_lost": 0,
            "enemy_units_killed_value": 0.0,
            "enemy_structures_destroyed_value": 0.0,
            "units_lost_value": 0.0,
            "structures_lost_value": 0.0,
        }
        # Reset bookkeeping and milestones used for reward shaping
        self._prev_metrics = None
        self._prev_minerals = None
        self._prev_time = None
        self._prev_score_cumulative = None
        self._milestone_16_achieved = False
        self._milestone_32_achieved = False
        self._milestone_48_achieved = False
        self._milestone_60_achieved = False