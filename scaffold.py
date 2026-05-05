from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
import sc2
import random
import numpy as np
import torch
import torch.nn.functional as F
try:
    from line_profiler import profile
except NameError:
    def profile(func):
        return func

# ---------------------------------------------------------------------------
# ACTION SPACE  (40 actions total)
#
# Design principles vs original:
#   1. NO redundant overlaps – economy management, worker assignment, and
#      overlord production are unified into single macro actions.
#   2. EXPLICIT unit production – the model picks the exact unit type.
#      train_zergling and train_baneling are separate so the model can
#      independently decide to mass lings vs convert them to banes.
#   3. RESEARCH covers all upgrade tiers, bundled per tech building.
#   4. SCOUT action exposed (the method already existed but had no action).
#   5. NYDUS / OVERSEER / CHANGELING absorbed into a single "specialist"
#      action rather than burning 4 slots for rarely-used niche moves.
#   6. DEFEND / ATTACK separated into "rally-defend", "attack-push", and
#      "harass-economy" – clear intent with distinct reward signals.
#   7. Action indices are STABLE: new actions are appended, not inserted,
#      so saved checkpoints remain forward-compatible.
#
# Index → intent
#  0  no-op
#  --- ECONOMY ---
#  1  manage_economy          (train drones + return idle workers to minerals)
#  2  saturate_gas            (build extractors + fill them to ideal)
#  3  ensure_supply           (train overlords until cap ≥ used+8)
#  4  expand                  (build hatchery at next free expansion)
#  5  train_queen             (one queen per idle hatchery with pool ready)
#  6  inject_larva            (queens inject all hatcheries with ≥25 energy)
#  7  spread_creep            (spare queens drop creep tumors; inject takes priority)
#  8  transfuse               (queens heal injured units)
#  --- T1 STRUCTURES ---
#  9  build_spawning_pool
# 10  build_roach_warren
# 11  build_baneling_nest
# 12  build_evolution_chamber
# 13  build_spine_crawler
# 14  build_spore_crawler
#  --- T2 STRUCTURES (require Lair) ---
# 15  morph_to_lair
# 16  build_hydralisk_den
# 17  build_spire
# 18  build_lurker_den
# 19  build_infestation_pit
#  --- T3 STRUCTURES (require Hive) ---
# 20  morph_to_hive
# 21  build_greater_spire
# 22  build_ultralisk_cavern
#  --- UNIT PRODUCTION ---
# 23  train_zergling          (mass zerglings from larva)
# 24  train_roach
# 25  train_t2_air            (mutalisk > corruptor based on availability)
# 26  train_hydralisk
# 27  train_lurker            (morphs hydras → lurkers)
# 28  train_t3_army           (ultralisk > brood lord > infestor in priority order)
#  --- UPGRADES (bundled per building) ---
# 29  research_pool_upgrades  (ling speed)
# 30  research_warren_upgrades (roach speed, then burrow)
# 31  research_evo_upgrades   (ground armor L1-L3, then melee L1-L3)
# 32  research_air_upgrades   (flyer attacks, air armor, hydra range+speed)
# 33  research_special        (neural parasite, baneling speed)
#  --- MILITARY ---
# 34  attack_push             (attack-move toward enemy base/units)
# 35  defend_base             (attack nearby threats near our base)
# 36  harass_economy          (zerglings/mutalisks hit enemy workers/minerals)
# 37  scout                   (idle zerglings/overlords explore expansions)
#  --- SPECIALIST ---
# 38  build_overseer          (morph idle overlord → overseer for detection)
#  --- T1 UNIT (appended for stability) ---
# 39  train_baneling          (morph zerglings → banelings)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ACTION PARAMETER SPECIFICATION
#
# Each action can take up to N_ACTION_PARAMS continuous parameters in [0, 1].
# The model outputs these via a sigmoid ParameterHead in staragent.py.
#
# param[0] = "quantity"   – how many units/structures to train/build (scaled per action)
# param[1] = "aggression" – for military actions: how aggressively to commit
# param[2] = "target_x"   – for attack_move: target x coordinate (normalized 0-1)
# param[3] = "target_y"   – for attack_move: target y coordinate (normalized 0-1)
# param[4] = "retreat_x"  – for attack_move: retreat location x (normalized 0-1)
# param[5] = "retreat_y"  – for attack_move: retreat location y (normalized 0-1)
#
# Actions with no parameters still receive the param vector; they ignore it.
# The mapping below documents which param index is meaningful per action.
#
# Index → (param_index, semantic, lo, hi)
#   lo/hi: the raw integer range that param[0] in [0,1] maps to.
# ---------------------------------------------------------------------------
N_ACTION_PARAMS = 6  # [quantity, aggression, target_x, target_y, retreat_x, retreat_y]

# For each action: tuple of (quantity_lo, quantity_hi) — aggression is always [0,1]
ACTION_PARAM_RANGES = {
    # economy
    1:  (1, 6),   # manage_economy: max_orders 1-6
    2:  (1, 3),   # saturate_gas: (unused qty, kept for uniform interface)
    3:  (1, 4),   # ensure_supply: max_overlords 1-4
    5:  (1, 3),   # train_queen: n 1-3
    # unit production
    23: (1, 6),   # train_zergling: n 1-6
    24: (1, 5),   # train_roach: n 1-5
    25: (1, 4),   # train_t2_air: n 1-4
    26: (1, 5),   # train_anti_air (hydralisk): n 1-5
    27: (1, 3),   # train_lurker: n 1-3
    28: (1, 3),   # train_t3_army: n 1-3
    39: (1, 4),   # train_baneling: n 1-4
    # military – param[1] = aggression controls force-commitment threshold
    34: (1, 1),   # attack_push: aggression only
    35: (1, 1),   # defend_base: aggression only
    36: (1, 8),   # harass_economy: max_units 1-8
}


class Scaffold(BotAI):
    def __init__(
        self,
        max_units_tracked: int = 200,
        reward_weights: dict | None = None,
        log_level: int = 0,
        min_attack_supply: float = 4.0,
        min_attack_units: int = 4,
        attack_defend_radius: float = 50.0,
        phase: int = 3,
    ):
        super().__init__()
        self.observation_space = ((4, 64, 64), 9, max_units_tracked * 11)
        self.total_actions = 41
        self._phase = phase
        self.max_units_tracked = max_units_tracked
        self.unit_command_uses_self_do = True

        self._reward_weights = {
            # ── Potential terms (smooth, fire every step) ──────────────────
            "phi_workers":        0.03,
            "phi_army_supply":    0.04,
            "phi_hatcheries":     0.30,
            "phi_income":         1.50,
            "phi_tech_level":     0.50,

            # ── Sparse event bonuses ────────────────────────────────────────
            "enemy_unit_kills":           0.30,
            "enemy_structures_destroyed": 0.80,
            "damage_dealt":               0.02,
            "expansion":                  1.00,
            "defense_kills":              0.50,

            # ── Penalty terms ───────────────────────────────────────────────
            "losses":             -0.10,
            "worker_loss":        -0.15,
            "supply_penalty":     -0.20,
            "low_worker_penalty": -0.10,

            # ── Milestone bonuses (one-time) ────────────────────────────────
            "worker_milestone_32": 0.50,
            "worker_milestone_60": 1.00,
        }
        self._log_level = log_level
        self._min_attack_supply = float(min_attack_supply)
        self._min_attack_units = int(min_attack_units)
        self._attack_defend_radius = float(attack_defend_radius)
        self.army_units = [
            UnitTypeId.ZERGLING, UnitTypeId.BANELING, UnitTypeId.HYDRALISK,
            UnitTypeId.MUTALISK, UnitTypeId.ROACH, UnitTypeId.CORRUPTOR,
            UnitTypeId.BROODLORD, UnitTypeId.ULTRALISK, UnitTypeId.INFESTOR,
            UnitTypeId.LURKERMP,
        ]
        self.worker_types = [UnitTypeId.DRONE, UnitTypeId.PROBE, UnitTypeId.SCV]
        self.structure_types = [
            UnitTypeId.SPAWNINGPOOL, UnitTypeId.HYDRALISKDEN, UnitTypeId.SPIRE,
            UnitTypeId.ROACHWARREN, UnitTypeId.BANELINGNEST, UnitTypeId.INFESTATIONPIT,
            UnitTypeId.GREATERSPIRE, UnitTypeId.SPINECRAWLER, UnitTypeId.SPORECRAWLER,
            UnitTypeId.EVOLUTIONCHAMBER, UnitTypeId.ULTRALISKCAVERN,
            UnitTypeId.NYDUSCANAL, UnitTypeId.LURKERDENMP,
            UnitTypeId.HATCHERY, UnitTypeId.LAIR, UnitTypeId.HIVE,
        ]
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
        self._reset_performance_accumulator()
        self._prev_metrics = None
        self._prev_minerals = None
        self._prev_collected_minerals = None
        self._prev_time = None
        self._prev_score_cumulative = None
        self.unit_tags_received_action = set()

    # =========================================================================
    # OBSERVATION
    # =========================================================================

    @profile
    def get_observation(self):
        target_h = self.observation_space[0][1]
        target_w = self.observation_space[0][2]

        height     = self.game_info.terrain_height.data_numpy.astype(np.float32)
        visibility = self.state.visibility.data_numpy.astype(np.float32)
        creep      = self.state.creep.data_numpy.astype(np.float32)
        pathing    = self.game_info.pathing_grid.data_numpy.astype(np.float32)

        height     = height / 255.0
        visibility = visibility / 2.0

        spatial_layers = np.stack(
            [
                self._resize_map(height,     target_h, target_w),
                self._resize_map(visibility, target_h, target_w),
                self._resize_map(creep,      target_h, target_w),
                self._resize_map(pathing,    target_h, target_w),
            ],
            axis=0,
        )

        gas_available      = min(self.vespene / 1000.0, 1.0)
        minerals_available = min(self.minerals / 1500.0, 1.0)
        supply_used        = self.supply_used / 200.0
        supply_cap         = self.supply_cap / 200.0
        game_time          = min(self.time / 1800.0, 1.0)
        worker_count       = self.units(UnitTypeId.DRONE).amount / 50.0
        army_count         = max(
            self.units.not_flying.amount - self.units(UnitTypeId.DRONE).amount, 0
        ) / 100.0
        drone_supply       = (self.units(UnitTypeId.DRONE).amount * 1) / 200.0

        income_rate = 0.0
        collected_minerals = float(getattr(self.state.score, "collected_minerals", 0.0))
        if (
            self._prev_collected_minerals is not None
            and self._prev_time is not None
            and self.time > self._prev_time
        ):
            dt = self.time - self._prev_time
            income_rate = max(
                0.0,
                min((collected_minerals - self._prev_collected_minerals) / max(dt, 0.1), 1000),
            ) / 0.5

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

        units = []
        for i in self.units:
            if len(units) >= self.max_units_tracked:
                break
            units.append(self.encode_unit(i))
        for i in self.enemy_units:
            if len(units) >= self.max_units_tracked:
                break
            units.append(self.encode_unit(i))
        return spatial_layers, resources, np.array(units)

    @staticmethod
    def _resize_map(layer: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if layer.shape == (target_h, target_w):
            return layer.astype(np.float32)
        tensor  = torch.from_numpy(layer).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=(target_h, target_w), mode="nearest")
        return resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    @staticmethod
    def _resize_map_channels(layer: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if layer.shape[1:] == (target_h, target_w):
            return layer.astype(np.float32)
        tensor  = torch.from_numpy(layer).unsqueeze(0)
        resized = F.interpolate(tensor, size=(target_h, target_w), mode="nearest")
        return resized.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_unit(self, unit: Unit) -> np.ndarray:
        return np.array([
            unit.type_id.value / len(UnitTypeId),
            unit.health / unit.health_max if unit.health_max > 0 else 0.0,
            unit.shield / unit.shield_max if unit.shield_max > 0 else 0.0,
            unit.energy / unit.energy_max if unit.energy_max > 0 else 0.0,
            unit.position.x / self.game_info.map_size[0],
            unit.position.y / self.game_info.map_size[1],
            float(unit.is_flying),
            float(unit.is_mine),
            float(unit.is_structure),
            float(unit.is_burrowed),
            unit.weapon_cooldown,
        ])

    # =========================================================================
    # ACTION DISPATCH
    # =========================================================================

    ACTION_NAMES = {
        0: "no-op", 1: "manage_economy", 2: "saturate_gas", 3: "ensure_supply",
        4: "expand", 5: "train_queen", 6: "inject_larva", 7: "spread_creep",
        8: "transfuse", 9: "build_spawning_pool", 10: "build_roach_warren",
        11: "build_baneling_nest", 12: "build_evolution_chamber", 13: "build_spine_crawler",
        14: "build_spore_crawler", 15: "morph_to_lair", 16: "build_hydralisk_den",
        17: "build_spire", 18: "build_lurker_den", 19: "build_infestation_pit",
        20: "morph_to_hive", 21: "build_greater_spire", 22: "build_ultralisk_cavern",
        23: "train_zergling", 24: "train_roach", 25: "train_t2_air", 26: "train_anti_air",
        27: "train_lurker", 28: "train_t3_army", 29: "research_pool_upgrades",
        30: "research_warren_upgrades", 31: "research_evo_upgrades", 32: "research_air_upgrades",
        33: "research_special", 34: "attack_push", 35: "defend_base", 36: "harass_economy",
        37: "scout", 38: "build_overseer", 39: "train_baneling",
        40: "retreat_to_location",
    }

    @staticmethod
    def _decode_param(raw: float, lo: int, hi: int) -> int:
        """Map a sigmoid output in [0, 1] to an integer in [lo, hi]."""
        return max(lo, min(hi, round(lo + raw * (hi - lo))))

    async def _execute_action(self, action_idx: int, action_params=None):
        """Execute action_idx, using action_params (length-N_ACTION_PARAMS float array/tensor
        in [0,1]) to modulate quantity and aggression.

        param[0] = quantity  (number of units/orders to issue, scaled per action)
        param[1] = aggression (0 = cautious, 1 = fully commit; used by military actions)
        param[2] = location_x (for attack/retreat to location: normalized x coord 0-1)
        param[3] = location_y (for attack/retreat to location: normalized y coord 0-1)
        param[4] = unused
        param[5] = unused
        """
        # Extract param values safely
        if action_params is not None:
            try:
                p0 = float(action_params[0])  # quantity
                p1 = float(action_params[1])  # aggression
                p2 = float(action_params[2])  # target_x
                p3 = float(action_params[3])  # target_y
                p4 = float(action_params[4])  # retreat_x
                p5 = float(action_params[5])  # retreat_y
            except Exception:
                p0, p1, p2, p3, p4, p5 = 0.5, 0.5, 0.0, 0.0, 0.0, 0.0
        else:
            p0, p1, p2, p3, p4, p5 = 0.5, 0.5, 0.0, 0.0, 0.0, 0.0

        def qty(lo, hi):
            return self._decode_param(p0, lo, hi)

        def _get_position_from_params(px: float, py: float) -> Point2 | None:
            """Convert normalized (0-1) params to map coordinates."""
            if px < 0.01 and py < 0.01:
                return None
            map_size = self.game_info.map_size
            x = px * map_size.width
            y = py * map_size.height
            return Point2((x, y))

        if action_idx == 0:   return False
        # ── ECONOMY ──────────────────────────────────────────────────────────
        if action_idx == 1:   return await self.manage_economy(max_orders=qty(1, 6))
        if action_idx == 2:   return await self.saturate_gas()
        if action_idx == 3:   return await self.ensure_supply(max_overlords=qty(1, 4))
        if action_idx == 4:   return await self.expand()
        if action_idx == 5:   return await self.train_queen(n=qty(1, 3))
        if action_idx == 6:   return await self.inject_larva()
        if action_idx == 7:   return await self.spread_creep()
        if action_idx == 8:   return await self.transfuse()
        # ── STRUCTURES ───────────────────────────────────────────────────────
        if action_idx == 9:   return await self.build_spawning_pool()
        if action_idx == 10:  return await self.build_roach_warren()
        if action_idx == 11:  return await self.build_baneling_nest()
        if action_idx == 12:  return await self.build_evolution_chamber()
        if action_idx == 13:  return await self.build_spine_crawler()
        if action_idx == 14:  return await self.build_spore_crawler()
        if action_idx == 15:  return await self.morph_to_lair()
        if action_idx == 16:  return await self.build_hydralisk_den()
        if action_idx == 17:  return await self.build_spire()
        if action_idx == 18:  return await self.build_lurker_den()
        if action_idx == 19:  return await self.build_infestation_pit()
        if action_idx == 20:  return await self.morph_to_hive()
        if action_idx == 21:  return await self.build_greater_spire()
        if action_idx == 22:  return await self.build_ultralisk_cavern()
        # ── UNIT PRODUCTION ──────────────────────────────────────────────────
        if action_idx == 23:  return await self.train_zergling(n=qty(1, 6))
        if action_idx == 24:  return await self.train_roach(n=qty(1, 5))
        if action_idx == 25:  return await self.train_t2_air(n=qty(1, 4))
        if action_idx == 26:  return await self.train_anti_air(n=qty(1, 5))
        if action_idx == 27:  return await self.train_lurker(n=qty(1, 3))
        if action_idx == 28:  return await self.train_t3_army(n=qty(1, 3))
        # ── UPGRADES ─────────────────────────────────────────────────────────
        if action_idx == 29:  return await self.research_pool_upgrades()
        if action_idx == 30:  return await self.research_warren_upgrades()
        if action_idx == 31:  return await self.research_evo_upgrades()
        if action_idx == 32:  return await self.research_air_upgrades()
        if action_idx == 33:  return await self.research_special()
        # ── MILITARY ─────────────────────────────────────────────────────────
        # aggression (p1) raises the effective attack threshold: low p1 = wait for
        # a larger army; high p1 = attack with fewer units.
        # location from p2,p3 (normalized 0-1 map coords); if both < 0.01, auto-detect
        if action_idx == 34:
            # Phase 1: hard-code attack to enemy bases, ignore params
            if self._phase == 1:
                return await self._attack_phase1()
            target = _get_position_from_params(p2, p3)
            return await self.attack_move(aggression=p1, target=target)
        if action_idx == 35:  return await self.defend_base(aggression=p1)
        if action_idx == 36:  return await self.harass_economy(max_units=qty(1, 8))
        if action_idx == 37:  return await self.scout()
        if action_idx == 38:  return await self.build_overseer()
        if action_idx == 39:  return await self.train_baneling(n=qty(1, 4))
        # Retreat to location then attack nearby (p2,p3 = x,y)
        if action_idx == 40:
            retreat_to = _get_position_from_params(p2, p3)
            return await self.attack_move(aggression=p1, retreat_to=retreat_to) if retreat_to else False
        return False

    # =========================================================================
    # ACTION MASK
    # =========================================================================

    async def get_action_mask(self) -> torch.Tensor:
        """Return a boolean mask where True = action is legal right now."""
        mask = torch.ones(self.total_actions, dtype=torch.bool)

        # ── phase-based action enablement ─────────────────────────────────────────
        # Phase 1 (8 actions): 0, 1, 3, 4, 9, 23, 34, 35
        # Phase 2 (15 actions): adds 2, 5, 6, 7, 11, 29, 39
        # Phase 3 (full): all actions
        valid_indices_phase1 = {0, 1, 3, 4, 9, 23, 34, 35}
        valid_indices_phase2 = valid_indices_phase1 | {2, 5, 6, 7, 11, 29, 39}
        if self._phase == 1:
            valid_indices = valid_indices_phase1
        elif self._phase == 2:
            valid_indices = valid_indices_phase2
        else:
            valid_indices = None  # full actions

        if valid_indices is not None:
            for i in range(self.total_actions):
                if i not in valid_indices:
                    mask[i] = False

        # ── pre-compute shared predicates ────────────────────────────────────
        has_pool          = self.structures(UnitTypeId.SPAWNINGPOOL).exists
        has_pool_ready    = self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists
        has_lair          = self.structures(UnitTypeId.LAIR).ready.exists
        has_hive          = self.structures(UnitTypeId.HIVE).ready.exists
        has_hydra_den_r   = self.structures(UnitTypeId.HYDRALISKDEN).ready.exists
        has_spire_r       = self.structures(UnitTypeId.SPIRE).ready.exists
        has_greater_spire = self.structures(UnitTypeId.GREATERSPIRE).exists
        has_greater_spire_r = self.structures(UnitTypeId.GREATERSPIRE).ready.exists
        has_roach_warren_r= self.structures(UnitTypeId.ROACHWARREN).ready.exists
        has_baneling_nest_r = self.structures(UnitTypeId.BANELINGNEST).ready.exists
        has_infestation_pit_r = self.structures(UnitTypeId.INFESTATIONPIT).ready.exists
        has_ultralisk_cavern_r = self.structures(UnitTypeId.ULTRALISKCAVERN).ready.exists
        has_lurker_den_r  = self.structures(UnitTypeId.LURKERDENMP).ready.exists
        has_evo_r         = self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready.exists
        has_hatchery      = self.townhalls.ready.exists
        has_extractor_r   = self.structures(UnitTypeId.EXTRACTOR).ready.exists
        has_workers       = self.workers.exists
        has_larva         = self.larva.exists
        has_queen         = self.units(UnitTypeId.QUEEN).ready.exists
        has_queen_energy  = self.units(UnitTypeId.QUEEN).filter(lambda q: q.energy >= 25).exists
        has_zergling      = self.units(UnitTypeId.ZERGLING).ready.exists
        has_hydra         = self.units(UnitTypeId.HYDRALISK).ready.exists
        has_corruptor     = self.units(UnitTypeId.CORRUPTOR).ready.exists
        has_overlord      = self.units(UnitTypeId.OVERLORD).idle.exists
        has_army          = self.units.of_type(set(self.army_units)).ready.exists
        army_count        = self.units.of_type(set(self.army_units)).ready.amount
        army_supply       = float(self.supply_army)
        has_mutalisk      = self.units(UnitTypeId.MUTALISK).ready.exists

        # Unclaimed geyser check (needed for saturate_gas)
        extractors_exist = self.structures(UnitTypeId.EXTRACTOR)
        has_unclaimed_geyser = any(
            not extractors_exist.closer_than(1.0, g)
            for th in self.townhalls.ready
            for g in self.vespene_geyser.closer_than(10, th)
        )
        # Under-saturated gas
        has_unsaturated_gas = any(
            int(getattr(e, "assigned_harvesters", 0)) < int(getattr(e, "ideal_harvesters", 3))
            for e in self.structures(UnitTypeId.EXTRACTOR).ready
        )

        # Enemy near base check (for defend/attack masks)
        enemy_close = False
        if self.enemy_units:
            base_pos = self.townhalls.center if self.townhalls else self.start_location
            if base_pos is not None:
                enemy_close = self.enemy_units.closer_than(self._attack_defend_radius, base_pos).exists

        # ── 0: no-op — always valid ───────────────────────────────────────────

        # ── 1: manage_economy ────────────────────────────────────────────────
        # Always valid — can always try to train a drone or return idle workers.

        # ── 2: saturate_gas ──────────────────────────────────────────────────
        # Valid if we can build a new extractor OR fill an existing one.
        can_build_extractor = (
            has_unclaimed_geyser and has_hatchery and has_workers
            and self.can_afford(UnitTypeId.EXTRACTOR)
        )
        can_fill_gas = has_unsaturated_gas and has_workers
        if not (can_build_extractor or can_fill_gas):
            mask[2] = False

        # ── 3: ensure_supply ─────────────────────────────────────────────────
        if not has_larva or self.supply_cap >= 200 or self.supply_left > 8:
            mask[3] = False

        # ── 4: expand ────────────────────────────────────────────────────────
        if (not self.can_afford(UnitTypeId.HATCHERY)) or (await self.get_next_expansion() is None):
            mask[4] = False

        # ── 5: train_queen ───────────────────────────────────────────────────
        if not has_pool_ready or not has_hatchery or not self.can_afford(UnitTypeId.QUEEN):
            mask[5] = False

        # ── 6: inject_larva ──────────────────────────────────────────────────
        if not has_queen_energy:
            mask[6] = False

        # ── 7: spread_creep ──────────────────────────────────────────────────
        # Only valid if there are more energy-ready queens than hatcheries
        # (so inject always takes priority).
        queens_with_energy = self.units(UnitTypeId.QUEEN).ready.filter(
            lambda q: q.energy >= 25
        ).amount
        spare_queens_for_creep = queens_with_energy > self.townhalls.ready.amount
        if not spare_queens_for_creep or not has_queen_energy:
            mask[7] = False

        # ── 8: transfuse ─────────────────────────────────────────────────────
        queen_can_transfuse = self.units(UnitTypeId.QUEEN).filter(lambda q: q.energy >= 50).exists
        injured             = self.units.filter(lambda u: u.is_mine and u.health < u.health_max * 0.9).exists
        if not queen_can_transfuse or not injured:
            mask[8] = False

        # ── 9: build_spawning_pool ───────────────────────────────────────────
        if has_pool or not has_hatchery or not self.can_afford(UnitTypeId.SPAWNINGPOOL):
            mask[9] = False

        # ── 10: build_roach_warren ───────────────────────────────────────────
        if (
            self.structures(UnitTypeId.ROACHWARREN).exists
            or not has_pool_ready
            or not self.can_afford(UnitTypeId.ROACHWARREN)
        ):
            mask[10] = False

        # ── 11: build_baneling_nest ──────────────────────────────────────────
        if (
            self.structures(UnitTypeId.BANELINGNEST).exists
            or not has_pool_ready
            or not self.can_afford(UnitTypeId.BANELINGNEST)
        ):
            mask[11] = False

        # ── 12: build_evolution_chamber ──────────────────────────────────────
        if (
            self.structures(UnitTypeId.EVOLUTIONCHAMBER).exists
            or not has_pool_ready
            or not self.can_afford(UnitTypeId.EVOLUTIONCHAMBER)
        ):
            mask[12] = False

        # ── 13: build_spine_crawler ──────────────────────────────────────────
        if (
            self.structures(UnitTypeId.SPINECRAWLER).exists
            or self.already_pending(UnitTypeId.SPINECRAWLER)
            or not has_hatchery or not has_workers
            or not self.can_afford(UnitTypeId.SPINECRAWLER)
            or not has_pool_ready
        ):
            mask[13] = False

        # ── 14: build_spore_crawler ──────────────────────────────────────────
        if (
            self.structures(UnitTypeId.SPORECRAWLER).exists
            or self.already_pending(UnitTypeId.SPORECRAWLER)
            or not has_hatchery or not has_workers
            or not self.can_afford(UnitTypeId.SPORECRAWLER)
        ):
            mask[14] = False

        # ── 15: morph_to_lair ────────────────────────────────────────────────
        if (
            has_lair or has_hive
            or not has_pool_ready
            or not self.can_afford(UnitTypeId.LAIR)
            or not self.structures(UnitTypeId.HATCHERY).ready.exists
        ):
            mask[15] = False

        # ── 16: build_hydralisk_den ──────────────────────────────────────────
        if (
            self.structures(UnitTypeId.HYDRALISKDEN).exists
            or not has_lair
            or not self.can_afford(UnitTypeId.HYDRALISKDEN)
        ):
            mask[16] = False

        # ── 17: build_spire ──────────────────────────────────────────────────
        if (
            self.structures(UnitTypeId.SPIRE).exists
            or self.structures(UnitTypeId.GREATERSPIRE).exists
            or not has_lair
            or not self.can_afford(UnitTypeId.SPIRE)
        ):
            mask[17] = False

        # ── 18: build_lurker_den ─────────────────────────────────────────────
        if (
            self.structures(UnitTypeId.LURKERDENMP).exists
            or not has_lair
            or not self.can_afford(UnitTypeId.LURKERDENMP)
        ):
            mask[18] = False

        # ── 19: build_infestation_pit ────────────────────────────────────────
        if (
            self.structures(UnitTypeId.INFESTATIONPIT).exists
            or not has_lair
            or not self.can_afford(UnitTypeId.INFESTATIONPIT)
        ):
            mask[19] = False

        # ── 20: morph_to_hive ────────────────────────────────────────────────
        if (
            has_hive
            or not has_lair
            or not self.can_afford(UnitTypeId.HIVE)
        ):
            mask[20] = False

        # ── 21: build_greater_spire ──────────────────────────────────────────
        if (
            has_greater_spire
            or not has_spire_r
            or not has_hive
            or not self.can_afford(UnitTypeId.GREATERSPIRE)
        ):
            mask[21] = False

        # ── 22: build_ultralisk_cavern ───────────────────────────────────────
        if (
            self.structures(UnitTypeId.ULTRALISKCAVERN).exists
            or not has_hive
            or not self.can_afford(UnitTypeId.ULTRALISKCAVERN)
        ):
            mask[22] = False

        # ── 23: train_t1_army (lings / banelings) ────────────────────────────
        # Valid if pool ready and larva available.
        if not has_pool_ready or not has_larva:
            mask[23] = False

        # ── 24: train_roach ──────────────────────────────────────────────────
        if not has_roach_warren_r or not has_larva or not self.can_afford(UnitTypeId.ROACH):
            mask[24] = False

        # ── 25: train_t2_air (mutalisk or corruptor) ─────────────────────────
        can_muta = has_spire_r and has_larva and self.can_afford(UnitTypeId.MUTALISK)
        can_corr = has_spire_r and has_larva and self.can_afford(UnitTypeId.CORRUPTOR)
        if not (can_muta or can_corr):
            mask[25] = False

        # ── 26: train_hydralisk ──────────────────────────────────────────────
        if not has_hydra_den_r or not has_larva or not self.can_afford(UnitTypeId.HYDRALISK):
            mask[26] = False

        # ── 27: train_lurker ─────────────────────────────────────────────────
        if not has_lurker_den_r or not has_hydra or not self.can_afford(UnitTypeId.LURKERMP):
            mask[27] = False

        # ── 28: train_t3_army (ultra/broodlord/infestor) ─────────────────────
        can_ultra  = has_ultralisk_cavern_r and has_larva and self.can_afford(UnitTypeId.ULTRALISK)
        can_bl     = has_greater_spire_r and has_corruptor and self.can_afford(UnitTypeId.BROODLORD)
        can_infest = has_infestation_pit_r and has_larva and self.can_afford(UnitTypeId.INFESTOR)
        if not (can_ultra or can_bl or can_infest):
            mask[28] = False

        # ── 29: research_pool_upgrades ───────────────────────────────────────
        can_ling_speed = (
            has_pool_ready
            and not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED)
            and self.can_afford(UpgradeId.ZERGLINGMOVEMENTSPEED)
        )
        if not can_ling_speed:
            mask[29] = False

        # ── 30: research_warren_upgrades ─────────────────────────────────────
        can_roach_speed = (
            has_roach_warren_r
            and not self.already_pending_upgrade(UpgradeId.GLIALRECONSTITUTION)
            and self.can_afford(UpgradeId.GLIALRECONSTITUTION)
        )
        can_burrow = (
            (has_lair or has_hive)
            and not self.already_pending_upgrade(UpgradeId.BURROW)
            and self.can_afford(UpgradeId.BURROW)
        )
        if not (can_roach_speed or can_burrow):
            mask[30] = False

        # ── 31: research_evo_upgrades ────────────────────────────────────────
        _evo_upgrades = [
            UpgradeId.ZERGGROUNDARMORSLEVEL1, UpgradeId.ZERGGROUNDARMORSLEVEL2,
            UpgradeId.ZERGGROUNDARMORSLEVEL3, UpgradeId.ZERGMELEEWEAPONSLEVEL1,
            UpgradeId.ZERGMELEEWEAPONSLEVEL2, UpgradeId.ZERGMELEEWEAPONSLEVEL3,
        ]
        can_evo = has_evo_r and any(
            not self.already_pending_upgrade(u) and self.can_afford(u)
            for u in _evo_upgrades
        )
        if not can_evo:
            mask[31] = False

        # ── 32: research_air_upgrades ────────────────────────────────────────
        can_flyer_atk = (
            has_spire_r
            and not self.already_pending_upgrade(UpgradeId.ZERGFLYERWEAPONSLEVEL1)
            and self.can_afford(UpgradeId.ZERGFLYERWEAPONSLEVEL1)
        )
        can_hydra_rng = (
            has_hydra_den_r
            and not self.already_pending_upgrade(UpgradeId.EVOLVEGROOVEDSPINES)
            and self.can_afford(UpgradeId.EVOLVEGROOVEDSPINES)
        )
        can_hydra_spd = (
            has_hydra_den_r
            and not self.already_pending_upgrade(UpgradeId.EVOLVEMUSCULARAUGMENTS)
            and self.can_afford(UpgradeId.EVOLVEMUSCULARAUGMENTS)
        )
        if not (can_flyer_atk or can_hydra_rng or can_hydra_spd):
            mask[32] = False

        # ── 33: research_special (neural, baneling speed) ────────────────────
        can_neural = (
            has_infestation_pit_r
            and not self.already_pending_upgrade(UpgradeId.NEURALPARASITE)
            and self.can_afford(UpgradeId.NEURALPARASITE)
        )
        can_bane_spd = (
            has_baneling_nest_r
            and not self.already_pending_upgrade(UpgradeId.CENTRIFICALHOOKS)
            and self.can_afford(UpgradeId.CENTRIFICALHOOKS)
        )
        if not (can_neural or can_bane_spd):
            mask[33] = False

        # ── 34: attack_push ──────────────────────────────────────────────────
        min_supply = max(0.0, self._min_attack_supply)
        min_units  = max(0, self._min_attack_units)
        if not has_army or (
            (army_supply < min_supply or army_count < min_units) and not enemy_close
        ):
            mask[34] = False

        # ── 35: defend_base ──────────────────────────────────────────────────
        if not has_army or not enemy_close:
            mask[35] = False

        # ── 36: harass_economy ───────────────────────────────────────────────
        # Valid if we have any zergling or mutalisk and know where the enemy is
        has_harasser = has_zergling or has_mutalisk
        has_harass_target = bool(
            self.enemy_units.of_type(set(self.worker_types)).exists
            or self.enemy_start_locations
        )
        if not has_harasser or not has_harass_target:
            mask[36] = False

        # ── 37: scout ────────────────────────────────────────────────────────
        if not has_zergling and not self.units(UnitTypeId.OVERLORD).idle.exists:
            mask[37] = False

        # ── 38: build_overseer ───────────────────────────────────────────────
        if (
            not has_overlord
            or not (has_lair or has_hive)
            or not self.can_afford(UnitTypeId.OVERSEER)
        ):
            mask[38] = False

        # ── 40: retreat_to_location ─────────────────────────────────────────
        # Requires army (like attack_move)
        min_supply = max(0.0, self._min_attack_supply)
        min_units = max(0, self._min_attack_units)
        if not has_army or (
            (army_supply < min_supply or army_count < min_units) and not enemy_close
        ):
            mask[40] = False

        return mask

    # =========================================================================
    # ECONOMY ACTIONS
    # =========================================================================

    async def manage_economy(self, max_orders: int = 6) -> bool:
        """Train drones up to ideal count (up to max_orders per call) and
        return idle workers to minerals.
        """
        acted = False
        ideal_workers = self.townhalls.amount * 16 + self.structures(UnitTypeId.EXTRACTOR).ready.amount * 3
        drones_to_train = min(
            max_orders,
            max(0, min(ideal_workers, 70) - self.workers.amount),
        )
        for _ in range(drones_to_train):
            larva = self._available_larva()
            if not larva or not self.can_afford(UnitTypeId.DRONE) or self.supply_left < 1:
                break
            self.do(larva.random.train(UnitTypeId.DRONE))
            acted = True

        gatherable = self.workers.filter(lambda w: not w.is_constructing_scv)
        if gatherable.idle and self.mineral_field:
            for w in list(gatherable.idle)[:max_orders]:
                self.do(w.gather(self.mineral_field.closest_to(w)))
                acted = True
        return acted

    async def saturate_gas(self) -> bool:
        """Build extractors on unclaimed geysers, then fill under-saturated ones.

        Replaces the original build_extractor + send_drone_gas + gather_vespene
        trio, which were three separate actions doing overlapping things.
        """
        acted = False

        # Build extractors on unclaimed geysers near ready townhalls
        if self.can_afford(UnitTypeId.EXTRACTOR):
            for th in self.townhalls.ready:
                for geyser in self.vespene_geyser.closer_than(10, th):
                    if self.structures(UnitTypeId.EXTRACTOR).closer_than(1.0, geyser):
                        continue
                    worker = self.select_build_worker(geyser.position)
                    if worker:
                        self.do(worker.build(UnitTypeId.EXTRACTOR, geyser))
                        acted = True
                        break  # one per call to avoid spending too many drones

        # Fill under-saturated extractors
        for extractor in self.structures(UnitTypeId.EXTRACTOR).ready:
            assigned = int(getattr(extractor, "assigned_harvesters", 0))
            ideal    = int(getattr(extractor, "ideal_harvesters", 3))
            if assigned >= ideal:
                continue
            # Prefer idle workers, fall back to any worker
            if self.workers.idle:
                worker = self.workers.idle.random
            elif self.workers:
                worker = self.workers.random
            else:
                break
            self.do(worker.gather(extractor))
            acted = True

        return acted

    async def ensure_supply(self, max_overlords: int = 4) -> bool:
        """Train overlords until supply headroom >= 8 or cap == 200.

        Replaces both the original train_overlord + ensure_overlord_buffer
        actions.  One action is enough; the mask only enables it when supply
        is actually tight.
        """
        if self.supply_cap >= 200:
            return False
        issued = 0
        while self.supply_left < 8 and self.supply_cap < 200 and issued < max_overlords:
            if not (self.larva and self.can_afford(UnitTypeId.OVERLORD)):
                break
            self.do(self.larva.random.train(UnitTypeId.OVERLORD))
            issued += 1
        return issued > 0

    async def expand(self) -> bool:
        if not self.can_afford(UnitTypeId.HATCHERY):
            return False
        if await self.get_next_expansion() is None:
            return False
        return bool(await self.expand_now())

    # =========================================================================
    # QUEEN ACTIONS  (unchanged logic, cleaned up)
    # =========================================================================

    async def train_queen(self, n: int = 1) -> bool:
        if not self.structures(UnitTypeId.SPAWNINGPOOL).ready or not self.can_afford(UnitTypeId.QUEEN):
            return False
        issued = 0
        for hatchery in self.townhalls.ready:
            if issued >= n:
                break
            if self.already_pending(UnitTypeId.QUEEN) >= self.townhalls.amount:
                break
            if not self.can_afford(UnitTypeId.QUEEN):
                break
            self.do(hatchery.train(UnitTypeId.QUEEN))
            issued += 1
        return issued > 0

    async def inject_larva(self) -> bool:
        queens = self.units(UnitTypeId.QUEEN).ready.filter(lambda q: q.energy >= 25)
        if not queens or not self.townhalls.ready:
            return False
        issued = False
        for i, hatch in enumerate(list(self.townhalls.ready)):
            if i >= queens.amount:
                break
            self.do(queens[i](AbilityId.EFFECT_INJECTLARVA, hatch))
            issued = True
        return issued

    async def spread_creep(self) -> bool:
        """Drop creep tumors using a fast numpy candidate search.

        Only queens with ≥25 energy that are NOT needed for inject are used,
        so inject always takes priority over spreading.
        """
        # Only use queens whose energy won't be needed for inject this step
        hatch_count = self.townhalls.ready.amount
        inject_queens = self.units(UnitTypeId.QUEEN).ready.filter(
            lambda q: q.energy >= 25
        ).amount
        # Queens beyond the hatchery count are free to spread creep
        free_queen_count = max(0, inject_queens - hatch_count)
        if free_queen_count == 0:
            return False

        creep_grid = self.state.creep
        if creep_grid is None:
            return False

        creep_arr = creep_grid.data_numpy  # shape (H, W), dtype uint8
        tumors = list(self.structures.of_type(
            {UnitTypeId.CREEPTUMOR, UnitTypeId.CREEPTUMORBURROWED}
        ))
        tumor_positions = np.array(
            [[t.position.x, t.position.y] for t in tumors], dtype=np.float32
        ) if tumors else None

        # Candidate cells: anywhere creep exists
        ys, xs = np.where(creep_arr > 0)
        if len(xs) == 0:
            return False

        acted = False
        free_queens = self.units(UnitTypeId.QUEEN).ready.filter(
            lambda q: q.energy >= 25
        )
        for queen in list(free_queens)[:free_queen_count]:
            qx, qy = queen.position.x, queen.position.y

            # Restrict candidates to a sensible radius around the queen
            dist_sq = (xs - qx) ** 2 + (ys - qy) ** 2
            nearby = (dist_sq >= 25) & (dist_sq <= 400)  # 5–20 tiles
            if not nearby.any():
                continue

            cxs, cys = xs[nearby], ys[nearby]

            # Filter out cells too close to existing tumors
            if tumor_positions is not None:
                pts = np.stack([cxs, cys], axis=1).astype(np.float32)
                # min distance to any tumor for each candidate
                diffs = pts[:, None, :] - tumor_positions[None, :, :]  # (N, T, 2)
                min_tumor_dist = np.sqrt((diffs ** 2).sum(axis=2)).min(axis=1)
                cxs = cxs[min_tumor_dist >= 8]
                cys = cys[min_tumor_dist >= 8]

            if len(cxs) == 0:
                continue

            # Pick the candidate closest to the queen
            d2 = (cxs - qx) ** 2 + (cys - qy) ** 2
            best = np.argmin(d2)
            target = Point2((float(cxs[best]), float(cys[best])))
            self.do(queen(AbilityId.BUILD_CREEPTUMOR_QUEEN, target))
            acted = True

        return acted

    async def transfuse(self) -> bool:
        queens = self.units(UnitTypeId.QUEEN).filter(lambda q: q.energy >= 50)
        if not queens:
            return False
        injured = self.units.filter(lambda u: u.is_mine and u.health < u.health_max * 0.9)
        if not injured:
            return False
        self.do(queens.first(AbilityId.TRANSFUSION_TRANSFUSION, injured.first))
        return True

    # =========================================================================
    # STRUCTURE BUILDERS  (unchanged logic)
    # =========================================================================

    async def build_spawning_pool(self) -> bool:
        if self.structures(UnitTypeId.SPAWNINGPOOL).exists or not self.townhalls.ready:
            return False
        if not self.can_afford(UnitTypeId.SPAWNINGPOOL):
            return False
        pos = self.townhalls.ready.first.position.towards(self.game_info.map_center, 5)
        return bool(await self.build(UnitTypeId.SPAWNINGPOOL, near=pos))

    async def build_roach_warren(self) -> bool:
        if self.structures(UnitTypeId.ROACHWARREN).exists or not self.townhalls.ready:
            return False
        if not self.structures(UnitTypeId.SPAWNINGPOOL).ready or not self.can_afford(UnitTypeId.ROACHWARREN):
            return False
        pos = self.townhalls.ready.first.position.towards(self.game_info.map_center, 5)
        return bool(await self.build(UnitTypeId.ROACHWARREN, near=pos))

    async def build_baneling_nest(self) -> bool:
        if self.structures(UnitTypeId.BANELINGNEST).exists or not self.townhalls.ready:
            return False
        if not self.structures(UnitTypeId.SPAWNINGPOOL).ready or not self.can_afford(UnitTypeId.BANELINGNEST):
            return False
        pos = self.townhalls.ready.first.position.towards(self.game_info.map_center, 5)
        return bool(await self.build(UnitTypeId.BANELINGNEST, near=pos))

    async def build_evolution_chamber(self) -> bool:
        if self.structures(UnitTypeId.EVOLUTIONCHAMBER).exists or not self.townhalls.ready:
            return False
        if not self.can_afford(UnitTypeId.EVOLUTIONCHAMBER):
            return False
        pos = self.townhalls.ready.first.position.towards(self.game_info.map_center, 5)
        return bool(await self.build(UnitTypeId.EVOLUTIONCHAMBER, near=pos))

    async def build_spine_crawler(self) -> bool:
        if self.structures(UnitTypeId.SPINECRAWLER).exists or self.already_pending(UnitTypeId.SPINECRAWLER):
            return False
        if not self.can_afford(UnitTypeId.SPINECRAWLER) or not self.townhalls.ready or not self.workers:
            return False
        pos = self.townhalls.ready.first.position.towards(self.game_info.map_center, 7)
        return bool(await self.build(UnitTypeId.SPINECRAWLER, near=pos))

    async def build_spore_crawler(self) -> bool:
        if self.structures(UnitTypeId.SPORECRAWLER).exists or self.already_pending(UnitTypeId.SPORECRAWLER):
            return False
        if not self.can_afford(UnitTypeId.SPORECRAWLER) or not self.townhalls.ready or not self.workers:
            return False
        pos = self.townhalls.ready.first.position.towards(self.game_info.map_center, 7)
        return bool(await self.build(UnitTypeId.SPORECRAWLER, near=pos))

    async def morph_to_lair(self) -> bool:
        hatcheries = self.structures(UnitTypeId.HATCHERY).ready
        if not hatcheries or self.structures(UnitTypeId.LAIR).exists:
            return False
        if not self.can_afford(UnitTypeId.LAIR):
            return False
        hatch = hatcheries.closest_to(self.start_location) if self.start_location else hatcheries.first
        self.do(hatch.build(UnitTypeId.LAIR))
        return True

    async def morph_to_hive(self) -> bool:
        """Morph a ready Lair into Hive.  Replaces the old morph_to_greater_spire
        misuse as T3 gate — Greater Spire has its own action."""
        lairs = self.structures(UnitTypeId.LAIR).ready
        if not lairs or self.structures(UnitTypeId.HIVE).exists:
            return False
        if not self.can_afford(UnitTypeId.HIVE):
            return False
        self.do(lairs.first.build(UnitTypeId.HIVE))
        return True

    async def build_hydralisk_den(self) -> bool:
        if self.structures(UnitTypeId.HYDRALISKDEN).exists or not self.structures(UnitTypeId.LAIR).ready:
            return False
        if not self.can_afford(UnitTypeId.HYDRALISKDEN):
            return False
        pos = self.structures(UnitTypeId.LAIR).ready.first.position.towards(self.game_info.map_center, 5)
        return bool(await self.build(UnitTypeId.HYDRALISKDEN, near=pos))

    async def build_spire(self) -> bool:
        if self.structures(UnitTypeId.SPIRE).exists or self.structures(UnitTypeId.GREATERSPIRE).exists:
            return False
        if not self.structures(UnitTypeId.LAIR).ready or not self.can_afford(UnitTypeId.SPIRE):
            return False
        pos = self.structures(UnitTypeId.LAIR).ready.first.position.towards(self.game_info.map_center, 5)
        return bool(await self.build(UnitTypeId.SPIRE, near=pos))

    async def build_lurker_den(self) -> bool:
        if self.structures(UnitTypeId.LURKERDENMP).exists or not self.structures(UnitTypeId.LAIR).ready:
            return False
        if not self.can_afford(UnitTypeId.LURKERDENMP):
            return False
        pos = self.structures(UnitTypeId.LAIR).ready.first.position.towards(self.game_info.map_center, 5)
        return bool(await self.build(UnitTypeId.LURKERDENMP, near=pos))

    async def build_infestation_pit(self) -> bool:
        if self.structures(UnitTypeId.INFESTATIONPIT).exists or not self.structures(UnitTypeId.LAIR).ready:
            return False
        if not self.can_afford(UnitTypeId.INFESTATIONPIT):
            return False
        pos = self.structures(UnitTypeId.LAIR).ready.first.position.towards(self.game_info.map_center, 5)
        return bool(await self.build(UnitTypeId.INFESTATIONPIT, near=pos))

    async def build_greater_spire(self) -> bool:
        if self.structures(UnitTypeId.GREATERSPIRE).exists:
            return False
        if not self.structures(UnitTypeId.SPIRE).ready or not self.structures(UnitTypeId.HIVE).ready:
            return False
        if not self.can_afford(UnitTypeId.GREATERSPIRE):
            return False
        spire = self.structures(UnitTypeId.SPIRE).ready.first
        self.do(spire.build(UnitTypeId.GREATERSPIRE))
        return True

    async def build_ultralisk_cavern(self) -> bool:
        if self.structures(UnitTypeId.ULTRALISKCAVERN).exists or not self.structures(UnitTypeId.HIVE).ready:
            return False
        if not self.can_afford(UnitTypeId.ULTRALISKCAVERN):
            return False
        pos = self.structures(UnitTypeId.HIVE).ready.first.position.towards(self.game_info.map_center, 5)
        return bool(await self.build(UnitTypeId.ULTRALISKCAVERN, near=pos))

    # =========================================================================
    # UNIT PRODUCTION — Tier-grouped
    # =========================================================================

    def _available_larva(self):
        """Return larva not already assigned this step."""
        return self.units(UnitTypeId.LARVA).filter(lambda l: l.tag not in self.unit_tags_received_action)

    async def train_zergling(self, n: int = 3) -> bool:
        """Train zerglings — pure mass."""
        if not self.structures(UnitTypeId.SPAWNINGPOOL).ready:
            return False
        issued = False
        for _ in range(max(1, n)):
            larva = self._available_larva()
            if not larva or not self.can_afford(UnitTypeId.ZERGLING) or self.supply_left < 1:
                break
            self.do(larva.random.train(UnitTypeId.ZERGLING))
            issued = True
        return issued

    async def train_baneling(self, n: int = 3) -> bool:
        """Morph zerglings into banelings."""
        if not self.structures(UnitTypeId.BANELINGNEST).ready:
            return False
        issued = False
        for _ in range(max(1, n)):
            lings = self.units(UnitTypeId.ZERGLING).ready
            if not lings or not self.can_afford(UnitTypeId.BANELING):
                break
            self.do(lings.random.train(UnitTypeId.BANELING))
            issued = True
        return issued

    async def train_roach(self, n: int = 3) -> bool:
        if not self.structures(UnitTypeId.ROACHWARREN).ready:
            return False
        issued = False
        for _ in range(max(1, n)):
            larva = self._available_larva()
            if not larva or not self.can_afford(UnitTypeId.ROACH) or self.supply_left < 2:
                break
            self.do(larva.random.train(UnitTypeId.ROACH))
            issued = True
        return issued

    async def train_t2_air(self, n: int = 2) -> bool:
        """Train mutalisks; fall back to corruptors only if mutalisk not affordable
        but spire is available (rare edge case - corruptor costs MORE).
        """
        if not self.structures(UnitTypeId.SPIRE).ready:
            return False
        issued = False
        for _ in range(max(1, n)):
            larva = self._available_larva()
            if not larva or self.supply_left < 2:
                break
            if self.can_afford(UnitTypeId.MUTALISK):
                self.do(larva.random.train(UnitTypeId.MUTALISK))
                issued = True
            elif self.can_afford(UnitTypeId.CORRUPTOR) and not self.can_afford(UnitTypeId.MUTALISK):
                self.do(larva.random.train(UnitTypeId.CORRUPTOR))
                issued = True
            else:
                break
        return issued

    async def train_anti_air(self, n: int = 3) -> bool:
        """Train hydralisks."""
        if not self.structures(UnitTypeId.HYDRALISKDEN).ready:
            return False
        issued = False
        for _ in range(max(1, n)):
            larva = self._available_larva()
            if not larva or not self.can_afford(UnitTypeId.HYDRALISK) or self.supply_left < 2:
                break
            self.do(larva.random.train(UnitTypeId.HYDRALISK))
            issued = True
        return issued

    async def train_lurker(self, n: int = 2) -> bool:
        """Morph hydralisk → lurker."""
        if not self.structures(UnitTypeId.LURKERDENMP).ready:
            return False
        issued = False
        for _ in range(max(1, n)):
            hydras = self.units(UnitTypeId.HYDRALISK).ready
            if not hydras or not self.can_afford(UnitTypeId.LURKERMP):
                break
            self.do(hydras.random.train(UnitTypeId.LURKERMP))
            issued = True
        return issued

    async def train_t3_army(self, n: int = 2) -> bool:
        """Train the best available T3 unit: ultralisk > brood lord > infestor."""
        issued = False
        for _ in range(max(1, n)):
            if (
                self.structures(UnitTypeId.ULTRALISKCAVERN).ready
                and self.larva and self.can_afford(UnitTypeId.ULTRALISK)
                and self.supply_left >= 6
            ):
                self.do(self.larva.random.train(UnitTypeId.ULTRALISK))
                issued = True
            elif (
                self.structures(UnitTypeId.GREATERSPIRE).ready
                and self.units(UnitTypeId.CORRUPTOR).ready
                and self.can_afford(UnitTypeId.BROODLORD)
            ):
                self.do(self.units(UnitTypeId.CORRUPTOR).ready.random.train(UnitTypeId.BROODLORD))
                issued = True
            elif (
                self.structures(UnitTypeId.INFESTATIONPIT).ready
                and self.larva and self.can_afford(UnitTypeId.INFESTOR)
                and self.supply_left >= 2
            ):
                self.do(self.larva.random.train(UnitTypeId.INFESTOR))
                issued = True
            else:
                break
        return issued

    # =========================================================================
    # BUNDLED RESEARCH
    # =========================================================================

    async def research_pool_upgrades(self) -> bool:
        """Zergling speed (only upgrade managed here for now; add attack
        upgrades if evo chamber is separate)."""
        if not self.structures(UnitTypeId.SPAWNINGPOOL).ready:
            return False
        pool = self.structures(UnitTypeId.SPAWNINGPOOL).ready.first
        if (
            not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED)
            and self.can_afford(UpgradeId.ZERGLINGMOVEMENTSPEED)
        ):
            self.do(pool.research(UpgradeId.ZERGLINGMOVEMENTSPEED))
            return True
        return False

    async def research_warren_upgrades(self) -> bool:
        """Roach speed, then burrow (at hive/lair townhall)."""
        issued = False
        if self.structures(UnitTypeId.ROACHWARREN).ready:
            warren = self.structures(UnitTypeId.ROACHWARREN).ready.first
            if (
                not self.already_pending_upgrade(UpgradeId.GLIALRECONSTITUTION)
                and self.can_afford(UpgradeId.GLIALRECONSTITUTION)
            ):
                self.do(warren.research(UpgradeId.GLIALRECONSTITUTION))
                issued = True
        if not issued and (self.structures(UnitTypeId.LAIR).ready or self.structures(UnitTypeId.HIVE).ready):
            th = self.townhalls.ready.first
            if (
                not self.already_pending_upgrade(UpgradeId.BURROW)
                and self.can_afford(UpgradeId.BURROW)
            ):
                self.do(th.research(UpgradeId.BURROW))
                issued = True
        return issued

    async def research_evo_upgrades(self) -> bool:
        """Ground upgrades from evolution chamber: armor L1→L2→L3, then melee L1→L2→L3."""
        if not self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready:
            return False
        evo = self.structures(UnitTypeId.EVOLUTIONCHAMBER).ready.first
        upgrade_priority = [
            UpgradeId.ZERGGROUNDARMORSLEVEL1,
            UpgradeId.ZERGGROUNDARMORSLEVEL2,
            UpgradeId.ZERGGROUNDARMORSLEVEL3,
            UpgradeId.ZERGMELEEWEAPONSLEVEL1,
            UpgradeId.ZERGMELEEWEAPONSLEVEL2,
            UpgradeId.ZERGMELEEWEAPONSLEVEL3,
        ]
        for upgrade in upgrade_priority:
            if (
                not self.already_pending_upgrade(upgrade)
                and self.can_afford(upgrade)
            ):
                self.do(evo.research(upgrade))
                return True
        return False

    async def research_air_upgrades(self) -> bool:
        """Flyer attacks (spire), hydra range + speed (hydra den)."""
        issued = False
        if self.structures(UnitTypeId.SPIRE).ready:
            spire = self.structures(UnitTypeId.SPIRE).ready.first
            if (
                not self.already_pending_upgrade(UpgradeId.ZERGFLYERWEAPONSLEVEL1)
                and self.can_afford(UpgradeId.ZERGFLYERWEAPONSLEVEL1)
            ):
                self.do(spire.research(UpgradeId.ZERGFLYERWEAPONSLEVEL1))
                issued = True
        if not issued and self.structures(UnitTypeId.HYDRALISKDEN).ready:
            den = self.structures(UnitTypeId.HYDRALISKDEN).ready.first
            for upgrade in [UpgradeId.EVOLVEGROOVEDSPINES, UpgradeId.EVOLVEMUSCULARAUGMENTS]:
                if (
                    not self.already_pending_upgrade(upgrade)
                    and self.can_afford(upgrade)
                ):
                    self.do(den.research(upgrade))
                    issued = True
                    break
        return issued

    async def research_special(self) -> bool:
        """Neural parasite (infestation pit) and baneling speed (baneling nest)."""
        if self.structures(UnitTypeId.INFESTATIONPIT).ready:
            pit = self.structures(UnitTypeId.INFESTATIONPIT).ready.first
            if (
                not self.already_pending_upgrade(UpgradeId.NEURALPARASITE)
                and self.can_afford(UpgradeId.NEURALPARASITE)
            ):
                self.do(pit.research(UpgradeId.NEURALPARASITE))
                return True
        if self.structures(UnitTypeId.BANELINGNEST).ready:
            nest = self.structures(UnitTypeId.BANELINGNEST).ready.first
            if (
                not self.already_pending_upgrade(UpgradeId.CENTRIFICALHOOKS)
                and self.can_afford(UpgradeId.CENTRIFICALHOOKS)
            ):
                self.do(nest.research(UpgradeId.CENTRIFICALHOOKS))
                return True
        return False

    # =========================================================================
    # MILITARY
    # =========================================================================

    async def attack_move(self, aggression: float = 0.5, target=None, retreat_to=None) -> bool:
        """Issue attack commands to the whole army.

        aggression in [0, 1]:
          0.0 → require full _min_attack_units before moving out
          1.0 → attack with as few as 1 unit

        target: optional Point2 location to attack. If None, auto-detects:
          visible enemy units → enemy structures → last known start location.

        retreat_to: optional Point2 location to move to first, then attack enemies near it.
        """
        effective_min = max(1, round(self._min_attack_units * (1.0 - aggression)))
        army = self.units.of_type(set(self.army_units)).ready
        if not army or len(army) < effective_min:
            return False

        if target is None:
            reference_point = retreat_to if retreat_to is not None else army.center
            if self.enemy_units:
                has_anti_air = any(u.can_attack_air for u in army)
                if has_anti_air:
                    target_unit = self.enemy_units.closest_to(reference_point)
                else:
                    ground_enemies = self.enemy_units.filter(lambda u: not u.is_flying)
                    target_unit = ground_enemies.closest_to(reference_point) if ground_enemies else None
                if target_unit is not None:
                    target = target_unit.position
            elif self.enemy_structures:
                townhall_types = {
                    UnitTypeId.HATCHERY, UnitTypeId.LAIR, UnitTypeId.HIVE,
                    UnitTypeId.NEXUS, UnitTypeId.COMMANDCENTER,
                    UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS,
                }
                key_targets = self.enemy_structures.of_type(townhall_types)
                target = (key_targets.first if key_targets else self.enemy_structures.first).position
            elif self.enemy_start_locations:
                target = self.enemy_start_locations[0]
            else:
                return False

        if retreat_to is not None and target is None:
            for unit in army:
                self.do(unit.move(retreat_to))
            return True

        # Check for flying enemies near target - ground-only units shouldn't attack them
        attack_target = target
        if self.enemy_units:
            enemies_near_target = self.enemy_units.closer_than(10, target)
            flying_near_target = enemies_near_target.filter(lambda u: u.is_flying)
            ground_near_target = enemies_near_target.filter(lambda u: not u.is_flying)

            # If there are ground enemies, prioritize them for ground armies
            if ground_near_target:
                attack_target = ground_near_target.closest_to(army.center).position
            elif flying_near_target and target:
                # Only attack flying if we have anti-air units
                has_anti_air = any(u.can_attack_air for u in army)
                if not has_anti_air:
                    # No anti-air, attack a structure instead or just move nearby
                    if self.enemy_structures:
                        attack_target = self.enemy_structures.closest_to(army.center).position
                    else:
                        attack_target = target  # fallback to original target
                # If has anti-air, proceed to attack flying target

        for unit in army:
            # Skip units that can't attack air if target has flying units nearby
            if attack_target and self.enemy_units:
                enemies_near_attack = self.enemy_units.closer_than(5, attack_target)
                flying_near = enemies_near_attack.filter(lambda u: u.is_flying)
                if flying_near and not unit.can_attack_air:
                    # This unit can't attack air, skip or attack ground instead
                    ground_near = enemies_near_attack.filter(lambda u: not u.is_flying)
                    if ground_near:
                        self.do(unit.attack(ground_near.closest_to(unit).position))
                    continue
            self.do(unit.attack(attack_target if attack_target else target))
        return True

    async def _attack_phase1(self) -> bool:
        """Phase 1 attack: target all known enemy bases, hybrid approach.

        Priority: visible enemies > structures > start locations.
        If no enemy info visible, attack closest start location.
        """
        army = self.units.of_type(set(self.army_units)).ready
        if not army or len(army) < self._min_attack_units:
            return False

        if self.enemy_units:
            has_anti_air = any(u.can_attack_air for u in army)
            if has_anti_air:
                target_unit = self.enemy_units.closest_to(army.center)
            else:
                ground_enemies = self.enemy_units.filter(lambda u: not u.is_flying)
                target_unit = ground_enemies.closest_to(army.center) if ground_enemies else None

            if target_unit is not None:
                attack_position = target_unit.position
                for unit in army:
                    self.do(unit.attack(attack_position))
                return True

        if self.enemy_structures:
            townhall_types = {
                UnitTypeId.HATCHERY, UnitTypeId.LAIR, UnitTypeId.HIVE,
                UnitTypeId.NEXUS, UnitTypeId.COMMANDCENTER,
                UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS,
            }
            key_structures = list(self.enemy_structures.of_type(townhall_types))
            other_structures = list(self.enemy_structures.filter(lambda u: u.type_id not in townhall_types))

            all_targets = []
            all_targets.extend(s.position for s in key_structures)
            all_targets.extend(s.position for s in other_structures)

            if all_targets:
                attack_position = army.center.closest(Point2(all_targets))
                for unit in army:
                    self.do(unit.attack(attack_position))
                return True

        if self.enemy_start_locations:
            attack_position = army.center.closest(self.enemy_start_locations)
            for unit in army:
                self.do(unit.attack(attack_position))
            return True

        return False

    async def defend_base(self, aggression: float = 0.5) -> bool:
        """Rally army to intercept near-base threats.

        aggression in [0, 1]:
          0.0 → only defend when enemies are very close (tight radius)
          1.0 → pre-emptively intercept at a wider radius
        """
        effective_radius = self._attack_defend_radius * (0.5 + aggression * 0.5)
        army = self.units.of_type(set(self.army_units)).ready
        if not army:
            return False

        base_pos = self.townhalls.center if self.townhalls else self.start_location
        if base_pos is None:
            return False

        if self.enemy_units:
            nearby = self.enemy_units.closer_than(effective_radius, base_pos)

            if nearby:
                flying_enemies = nearby.filter(lambda u: u.is_flying)
                ground_enemies = nearby.filter(lambda u: not u.is_flying)

                for unit in army:
                    target = None

                    if unit.can_attack_air and flying_enemies:
                        target = flying_enemies.closest_to(unit)
                    elif ground_enemies:
                        target = ground_enemies.closest_to(unit)

                    if target:
                        self.do(unit.attack(target))

                return True

        # No enemy nearby — action had no meaningful effect
        return False

    async def harass_economy(self, max_units: int = 8) -> bool:
        harassers = self.units(UnitTypeId.ZERGLING).ready
        if not harassers:
            harassers = self.units(UnitTypeId.MUTALISK).ready
        if not harassers:
            return False

        # Primary: visible enemy workers
        enemy_workers = self.enemy_units.of_type(set(self.worker_types))
        if enemy_workers:
            target = enemy_workers.closest_to(harassers.center).position
        # Secondary: mineral fields near known enemy base (harassers will auto-attack workers on arrival)
        elif self.enemy_start_locations:
            base = self.enemy_start_locations[0]
            nearby_minerals = self.mineral_field.closer_than(15, base)
            target = nearby_minerals.center if nearby_minerals else base
        else:
            return False

        for unit in list(harassers)[: min(int(max_units), harassers.amount)]:
            self.do(unit.attack(target))
        return True

    async def scout(self) -> bool:
        """Send an idle zergling (or overlord) to an unexplored expansion."""
        unexplored = [
            p for p in self.expansion_locations
            if self.state.visibility[int(p.x), int(p.y)] == 0
        ]
        if not unexplored:
            return False

        scouts = self.units(UnitTypeId.ZERGLING).idle
        if not scouts:
            scouts = self.units(UnitTypeId.OVERLORD).idle
        if not scouts:
            return False

        target = min(unexplored, key=lambda p: scouts.center.distance_to(p))
        self.do(scouts.random.move(target))
        return True

    async def build_overseer(self) -> bool:
        if not self.can_afford(UnitTypeId.OVERSEER):
            return False
        if not self.structures(UnitTypeId.LAIR).ready and not self.structures(UnitTypeId.HIVE).ready:
            return False
        overlords = self.units(UnitTypeId.OVERLORD).idle
        if not overlords:
            return False
        self.do(overlords.first(AbilityId.MORPH_OVERSEER))
        return True

    # =========================================================================
    # REWARD / METRICS  (unchanged from original)
    # =========================================================================

    def _collect_metrics(self) -> dict:
        score = self.state.score
        army_count   = sum(self.units(u).amount for u in self.army_units)
        enemy_kills  = (float(getattr(score, "killed_minerals_units", 0.0))
                        + float(getattr(score, "killed_vespene_units", 0.0)))
        structure_kills = (float(getattr(score, "killed_minerals_structures", 0.0))
                           + float(getattr(score, "killed_vespene_structures", 0.0)))
        lost_army    = (float(getattr(score, "lost_minerals_army", 0.0))
                        + float(getattr(score, "lost_vespene_army", 0.0)))
        lost_workers = (float(getattr(score, "lost_minerals_economy", 0.0))
                        + float(getattr(score, "lost_vespene_economy", 0.0)))
        hatcheries   = float(self.townhalls.amount)
        supply_used  = self.supply_used
        supply_cap   = self.supply_cap

        income_rate = 0.0
        collected_minerals = float(getattr(score, "collected_minerals", 0.0))
        if (
            self._prev_collected_minerals is not None
            and self._prev_time is not None
            and self.time > self._prev_time
        ):
            dt = self.time - self._prev_time
            income_rate = max(
                0.0, min((collected_minerals - self._prev_collected_minerals) / max(dt, 0.1), 1000) / 0.5
            )

        if self.structures(UnitTypeId.HIVE).ready:
            tech_level = 2.0
        elif self.structures(UnitTypeId.LAIR).ready:
            tech_level = 1.0
        else:
            tech_level = 0.0

        enemy_near_base = 0.0
        if self.townhalls and self.enemy_units:
            for th in self.townhalls:
                enemy_near_base += float(self.enemy_units.closer_than(self._attack_defend_radius, th).amount)

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
            "queen_count": float(self.units(UnitTypeId.QUEEN).ready.amount),
            "income_rate": income_rate,
            "tech_level": tech_level,
            "total_damage_dealt": float(getattr(score, "total_damage_dealt_life", 0.0))
                                  + float(getattr(score, "total_damage_dealt_shields", 0.0)),
            "enemy_near_base": enemy_near_base,
        }

    def _potential(self, metrics: dict) -> float:
        w = self._reward_weights
        workers     = min(metrics.get("workers", 0), 60)
        army_supply = max(min(metrics.get("supply_used", 0) - workers, 100), 0)
        hatcheries  = min(metrics.get("hatcheries", 0), 6)
        income      = min(metrics.get("income_rate", 0), 1.0)
        tech        = metrics.get("tech_level", 0)
        enemy_near  = min(metrics.get("enemy_near_base", 0), 30)
        base_safety = max(0, 30 - enemy_near) / 30.0
        return (
            w["phi_workers"]     * workers
            + w["phi_army_supply"] * army_supply
            + w["phi_hatcheries"]  * hatcheries
            + w["phi_income"]      * income
            + w["phi_tech_level"]  * tech
            + w["defense_kills"]   * base_safety
        )

    def _compute_step_reward(self, action_succeeded: bool, action_idx: int = -1) -> float:
        current = self._collect_metrics()
        self._update_performance_accumulator(current)
        self._sync_cumulative_score_fallback()

        if self._prev_metrics is None:
            self._prev_metrics          = current
            self._prev_minerals         = self.minerals
            self._prev_time             = self.time
            self._prev_collected_minerals = float(getattr(self.state.score, "collected_minerals", 0.0))
            return 0.0

        reward = self._potential(current) - self._potential(self._prev_metrics)

        score = self.state.score
        curr_damage = (float(getattr(score, "total_damage_dealt_life", 0.0))
                       + float(getattr(score, "total_damage_dealt_shields", 0.0)))
        prev_damage = self._prev_metrics.get("total_damage_dealt", 0.0)
        damage_delta = max(0.0, curr_damage - prev_damage)
        reward += self._reward_weights.get("damage_dealt", 0.0) * damage_delta

        self._prev_metrics            = current
        self._prev_minerals           = self.minerals
        self._prev_time               = self.time
        self._prev_collected_minerals = float(getattr(self.state.score, "collected_minerals", 0.0))

        return float(max(-3.0, min(3.0, reward)))

    def _sync_cumulative_score_fallback(self):
        score = self.state.score
        current_score_totals = {
            "enemy_units_killed_value": (float(getattr(score, "killed_minerals_units", 0.0))
                                         + float(getattr(score, "killed_vespene_units", 0.0))),
            "enemy_structures_destroyed_value": (float(getattr(score, "killed_minerals_structures", 0.0))
                                                  + float(getattr(score, "killed_vespene_structures", 0.0))),
            "units_lost_value": (float(getattr(score, "lost_minerals_army", 0.0))
                                 + float(getattr(score, "lost_vespene_army", 0.0))
                                 + float(getattr(score, "lost_minerals_economy", 0.0))
                                 + float(getattr(score, "lost_vespene_economy", 0.0))),
            "structures_lost_value": (float(getattr(score, "lost_minerals_technology", 0.0))
                                      + float(getattr(score, "lost_vespene_technology", 0.0))),
        }
        if self._prev_score_cumulative is None:
            self._prev_score_cumulative = current_score_totals
            return
        for key, total in current_score_totals.items():
            prev_total = float(self._prev_score_cumulative.get(key, 0.0))
            delta = max(0.0, total - prev_total)
            self._cumulative_stats[key] = float(self._cumulative_stats.get(key, 0.0)) + delta
        self._prev_score_cumulative = current_score_totals

    # =========================================================================
    # LOGGING / PERFORMANCE HELPERS  (unchanged)
    # =========================================================================

    def _log_step(self, iteration: int, action: int, succeeded: bool, reward: float):
        if self._log_level < 1:
            return
        print(f"[Scaffold] iter={iteration} action={action} ok={succeeded} r={reward:.4f}")

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
        print(f"[Scaffold] train loss: total={total_loss:.4f} rl={rl_loss:.4f} critic={critic_loss:.4f}")

    def _log_entropy(self, entropy: float):
        if self._log_level < 2:
            return
        print(f"[Scaffold] entropy: {entropy:.4f}")

    def _get_under_saturated_extractor(self):
        ready = self.structures(UnitTypeId.EXTRACTOR).ready
        if not ready:
            return None
        candidates = [
            (int(getattr(e, "assigned_harvesters", 0)), e)
            for e in ready
            if int(getattr(e, "assigned_harvesters", 0)) < max(1, int(getattr(e, "ideal_harvesters", 3)))
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda p: p[0])[0][1]

    def get_performance_metrics(self) -> dict:
        self._sync_cumulative_score_fallback()
        score = self.state.score

        averages = (
            {key: total / self._perf_steps for key, total in self._perf_accum.items()}
            if self._perf_steps > 0
            else {key: 0.0 for key in self._perf_accum}
        )

        return {
            "economic": {
                "mineral_collection_efficiency": score.collected_minerals / max(score.spent_minerals, 1),
                "vespene_collection_efficiency": score.collected_vespene / max(score.spent_vespene, 1),
                "idle_worker_time": score.idle_worker_time,
                "idle_production_time": score.idle_production_time,
            },
            "military": {
                "total_damage_dealt": score.total_damage_dealt_life + score.total_damage_dealt_shields,
                "total_damage_taken": score.total_damage_taken_life + score.total_damage_taken_shields,
                "damage_ratio": (score.total_damage_dealt_life + score.total_damage_dealt_shields)
                                / max(score.total_damage_taken_life + score.total_damage_taken_shields, 1),
                "kill_value_ratio": score.killed_value_units / max(
                    score.lost_minerals_army + score.lost_vespene_army
                    + score.lost_minerals_economy + score.lost_vespene_economy, 1
                ),
            },
            "resources": {
                "total_resources_collected": score.collected_minerals + score.collected_vespene,
                "total_resources_spent": score.spent_minerals + score.spent_vespene,
                "resource_spending_rate": (score.spent_minerals + score.spent_vespene)
                                          / max(score.collected_minerals + score.collected_vespene, 1),
            },
            "production": {
                "total_unit_value": score.total_value_units,
                "total_structure_value": score.total_value_structures,
                "total_value_created": score.total_value_units + score.total_value_structures,
                "value_lost_units": (score.lost_minerals_army + score.lost_vespene_army
                                     + score.lost_minerals_economy + score.lost_vespene_economy),
                "value_lost_structures": score.lost_minerals_technology + score.lost_vespene_technology,
                "net_value_retained": (
                    (score.total_value_units + score.total_value_structures)
                    - ((score.lost_minerals_army + score.lost_vespene_army
                        + score.lost_minerals_economy + score.lost_vespene_economy)
                       + (score.lost_minerals_technology + score.lost_vespene_technology))
                ),
            },
            "composition": {
                "workers": self.workers.amount,
                "army_count": self.units.amount - self.workers.amount,
                "structure_count": self.structures.amount,
                "supply_army": self.supply_army,
                "supply_workers": self.supply_workers,
                "supply_economy": score.food_used_economy,
                "supply_technology": score.food_used_technology,
            },
            "averages": averages,
            "game_time": self.time,
            "cumulative": self._cumulative_stats.copy(),
        }

    # =========================================================================
    # EVENT HOOKS  (unchanged)
    # =========================================================================

    async def on_unit_created(self, unit: Unit):
        await super().on_unit_created(unit)
        if unit.type_id in self.worker_types:
            self._cumulative_stats["workers_created"] += 1
        elif unit.type_id in self.army_units:
            self._cumulative_stats["army_units_created"] += 1

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

    def _reset_performance_accumulator(self):
        self._perf_accum = {
            "workers": 0.0, "army": 0.0, "minerals": 0.0, "gas": 0.0,
            "hatcheries": 0.0, "supply_used": 0.0, "supply_cap": 0.0,
            "structures": 0.0, "queen_count": 0.0, "income_rate": 0.0,
            "tech_level": 0.0,
        }
        self._perf_steps = 0

    def _update_performance_accumulator(self, metrics: dict):
        if not hasattr(self, "_perf_accum"):
            self._reset_performance_accumulator()
        for key in self._perf_accum:
            self._perf_accum[key] += float(metrics.get(key, 0.0))
        self._perf_steps += 1

    def reset_cumulative_stats(self):
        self._cumulative_stats = {
            "workers_created": 0, "army_units_created": 0, "structures_built": 0,
            "enemy_units_killed": 0, "enemy_structures_destroyed": 0,
            "units_lost": 0, "structures_lost": 0,
            "enemy_units_killed_value": 0.0, "enemy_structures_destroyed_value": 0.0,
            "units_lost_value": 0.0, "structures_lost_value": 0.0,
        }
        self._reset_performance_accumulator()
        self._prev_metrics            = None
        self._prev_minerals           = None
        self._prev_collected_minerals = None
        self._prev_time               = None
        self._prev_score_cumulative   = None
        self._milestone_32_achieved   = False
        self._milestone_60_achieved   = False