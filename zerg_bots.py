from scaffold import Scaffold
from sc2.ids.unit_typeid import UnitTypeId
import torch

# --- CheeseZergBot: Fast pool, early lings ---
class CheeseZergBot(Scaffold):
    def __init__(self, action_interval=3):
        super().__init__()
        self.action_interval = action_interval
        self._last_known_enemy_pos = None
        self._pool_started = False
        self._rushed = False

    async def on_step(self, iteration: int):
        """
        Cheese strategy: fast spawning pool into zergling rush.
        Priority ladder each tick:
          1. Always keep supply capped with overlords.
          2. Get spawning pool ASAP (before drones).
          3. Once pool is up, mass zerglings and attack.
          4. Inject larva to keep production flowing.
          5. If rush fails, transition to roaches as fallback.
        """
        if iteration % self.action_interval != 0:
            return

        # 1. Supply: build overlord if close to cap
        if self.supply_left < 2 and self.supply_cap < 200:
            await self._execute_action(5)  # train_overlord
            return

        # 2. Spawning pool first — hold drones until it's started
        if not self._pool_started:
            success = await self._execute_action(1)  # build_spawning_pool
            if success:
                self._pool_started = True
            else:
                await self._execute_action(4)  # train one drone to keep mining
            return

        # 3. Inject larva whenever possible for max production
        await self._execute_action(19)  # inject_larva

        # 4. Once pool exists, pump zerglings and attack
        spawning_pool_ready = self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists
        if spawning_pool_ready:
            # Research zergling speed if we haven't yet
            await self._execute_action(22)  # research_zergling_speed

            # Keep training lings
            await self._execute_action(2)  # train_zerglings(6)

            # Rush if we have enough units
            army_size = self.units(UnitTypeId.ZERGLING).amount
            if army_size >= 6 and not self._rushed:
                self._rushed = True
                await self._execute_action(3)  # attack_move
            elif self._rushed:
                # Keep attacking with whatever we have
                await self._execute_action(3)

        # 5. Fallback: if rush failed and we have few units, pivot to roaches
        if self._rushed:
            enemy_structures = self.enemy_structures
            if not enemy_structures and self.units(UnitTypeId.ZERGLING).amount < 4:
                # Rush failed — build roach warren as plan B
                await self._execute_action(10)  # build_roach_warren
                await self._execute_action(11)  # train_roach

    async def _execute_action(self, action_idx: int):
        if action_idx == 0:
            return False
        if action_idx == 1:
            return await self.build_spawning_pool()
        if action_idx == 2:
            return await self.train_zerglings(6)
        if action_idx == 3:
            return await self.attack_move(target=self._get_attack_target())
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
            return await self.train_baneling(2)
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
        return False

    def _get_attack_target(self):
        if self.enemy_units:
            return self.enemy_units.closest_to(self.start_location).position
        return self.enemy_start_locations[0]


# --- ReactiveZergBot: Defensive, counterattacks ---
class ReactiveZergBot(Scaffold):
    def __init__(self, action_interval=4):
        super().__init__()
        self.action_interval = action_interval
        self._last_known_enemy_pos = None
        self._under_attack = False
        self._counterattack_threshold = 15  # army size before pushing out

    async def on_step(self, iteration: int):
        """
        Reactive strategy: economy-first, defend with static defense and units,
        then counterattack once we have a critical mass.
        Priority ladder each tick:
          1. Keep supply open.
          2. Grow economy (drones up to 20 before heavy military).
          3. Detect incoming attacks — deploy spine crawlers + zerglings to defend.
          4. Inject larva and spread creep for map control.
          5. When army is large enough, counterattack and press advantage.
          6. Respond to air threats with spore crawlers / hydralisks.
        """
        if iteration % self.action_interval != 0:
            return

        # Detect if we are under attack
        self._under_attack = self.enemy_units.closer_than(30, self.start_location).exists

        # 1. Supply management
        if self.supply_left < 3 and self.supply_cap < 200:
            await self._execute_action(5)  # train_overlord
            return

        # 2. Economy: prioritize drones early
        drone_count = self.units(UnitTypeId.DRONE).amount
        if drone_count < 20 and not self._under_attack:
            await self._execute_action(1)  # train_drones
            return

        # 3. Static defense when under attack
        if self._under_attack:
            await self._execute_action(17)  # build_spine_crawler
            await self._execute_action(2)   # train_zerglings for defense
            await self._execute_action(29)  # focus_fire on attackers

            # Air threat detection: check for enemy air units
            enemy_air = [u for u in self.enemy_units if u.is_flying]
            if enemy_air:
                await self._execute_action(18)  # build_spore_crawler
                await self._execute_action(6)   # train_anti_air (hydralisks)

        # 4. Inject larva and spread creep each step
        await self._execute_action(19)  # inject_larva
        await self._execute_action(20)  # spread_creep

        # 5. Mid-game military build-up
        spawning_pool_ready = self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists
        if not spawning_pool_ready:
            await self._execute_action(1)  # train_drones (also builds pool via scaffold)

        roach_warren_ready = self.structures(UnitTypeId.ROACHWARREN).ready.exists
        if not roach_warren_ready and drone_count >= 16:
            await self._execute_action(10)  # build_roach_warren

        if roach_warren_ready:
            await self._execute_action(11)  # train_roach
            await self._execute_action(24)  # research_roach_speed

        # 6. Counterattack when army is large enough
        army = self.units.filter(lambda u: u.can_attack and not u.is_structure)
        if army.amount >= self._counterattack_threshold and not self._under_attack:
            await self._execute_action(3)   # rally_army
            await self._execute_action(4)   # attack_move
        elif self._under_attack and army.amount > 8:
            # Smaller counterattack to relieve pressure
            await self._execute_action(4)   # attack_move after defending

    async def _execute_action(self, action_idx: int):
        if action_idx == 0:
            return False
        if action_idx == 1:
            return await self.train_drones(1)
        if action_idx == 2:
            return await self.train_zerglings(4)
        if action_idx == 3:
            return await self.rally_army()
        if action_idx == 4:
            return await self.attack_move(target=self._get_attack_target())
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
            return await self.train_baneling(2)
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
        return False

    def _get_attack_target(self):
        if self.enemy_units:
            return self.enemy_units.closest_to(self.start_location).position
        return self.enemy_start_locations[0]


# --- HarassZergBot: Hit and run ---
class HarassZergBot(Scaffold):
    def __init__(self, action_interval=3):
        super().__init__()
        self.action_interval = action_interval
        self._last_known_enemy_pos = None
        self._harassing = False
        self._retreat_health_threshold = 0.4  # retreat if avg HP < 40%

    async def on_step(self, iteration: int):
        """
        Harass strategy: fast mutalisk / zergling hit-and-run attacks.
        Priority ladder each tick:
          1. Keep supply open.
          2. Build economy to support constant wave production.
          3. Tech to spire for mutalisks ASAP.
          4. Send small waves to harass enemy workers/structures.
          5. Retreat when units are low HP, regroup, then strike again.
          6. Research upgrades to keep waves threatening.
        """
        if iteration % self.action_interval != 0:
            return

        # 1. Supply management
        if self.supply_left < 3 and self.supply_cap < 200:
            await self._execute_action(5)  # train_overlord
            return

        # 2. Economy: enough drones to sustain waves
        drone_count = self.units(UnitTypeId.DRONE).amount
        if drone_count < 16:
            await self._execute_action(4)  # train_drones
            return

        # 3. Inject larva for constant unit production
        await self._execute_action(19)  # inject_larva

        # 4. Tech path: Spawning Pool -> Spire -> Mutalisks
        spawning_pool_ready = self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists
        if not spawning_pool_ready:
            await self._execute_action(1)   # train_zerglings triggers pool build in scaffold
            return

        spire_exists = self.structures(UnitTypeId.SPIRE).exists
        if not spire_exists:
            await self._execute_action(9)   # build_spire
            # While teching, send ling waves to buy time
            await self._execute_action(1)   # train_zerglings
            return

        spire_ready = self.structures(UnitTypeId.SPIRE).ready.exists
        if spire_ready:
            await self._execute_action(7)   # train_flying_unit (mutalisks)
            await self._execute_action(26)  # research_flyer_attacks

        # Also keep a small zergling escort for ground harassment
        await self._execute_action(1)       # train_zerglings(4)
        await self._execute_action(22)      # research_zergling_speed

        # 5. Harass decision: attack or retreat based on unit health
        flying_units = self.units(UnitTypeId.MUTALISK)
        ground_units = self.units(UnitTypeId.ZERGLING)
        harass_force = flying_units + ground_units

        if harass_force.amount >= 4:
            # Check average HP ratio of the harass force
            avg_hp_ratio = (
                sum(u.health_percentage for u in harass_force) / harass_force.amount
                if harass_force.amount > 0 else 1.0
            )

            if avg_hp_ratio < self._retreat_health_threshold:
                # Units are battered — pull back and regroup
                self._harassing = False
                await self._execute_action(27)  # retreat
                await self._execute_action(28)  # regroup
            else:
                # Units are healthy enough — go harass
                self._harassing = True
                await self._execute_action(2)   # attack_move toward enemy
                await self._execute_action(29)  # focus_fire priority targets
        else:
            # Not enough units yet — rally and wait for more
            self._harassing = False
            await self._execute_action(3)       # rally_army

    async def _execute_action(self, action_idx: int):
        if action_idx == 0:
            return False
        if action_idx == 1:
            return await self.train_zerglings(4)
        if action_idx == 2:
            return await self.attack_move(target=self._get_attack_target())
        if action_idx == 3:
            return await self.rally_army()
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
            return await self.train_baneling(2)
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
        return False

    def _get_attack_target(self):
        if self.enemy_units:
            return self.enemy_units.closest_to(self.start_location).position
        return self.enemy_start_locations[0]