from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import gymnasium

from fle.env import FactorioInstance
from fle.env.entities import Position, Direction, BoundingBox
from fle.env.game_types import Prototype, Resource

from fle.env.atomic_env.constants import (
    ACTION_NOOP, ACTION_PLACE_DRILL, ACTION_PLACE_POLE,
    N_ACTION_TYPES, GRID_SIZE, GRID_CENTER,
    CH_COAL_RESOURCE, CH_DRILLS, CH_POLES, CH_PLAYER,
    N_CHANNELS, N_FEATURES,
    FEAT_MINERS_PLACED, FEAT_MINERS_ON_COAL, FEAT_POLES_PLACED, FEAT_COAL_TOTAL,
    N_DIRECTIONS, DIRECTION_MAP,
    SHAPED_DRILL_ON_COAL_REWARD,
)


class CoalMiningAtomicEnv(gymnasium.Env):
    """
    Discrete-action RL environment for the coal mining task.

    Grid is centred on the coal patch (fixed reference per episode) rather
    than the player, so the agent always sees coal in the same grid position.

    Action space:
        MultiDiscrete([N_ACTION_TYPES, GRID_SIZE, GRID_SIZE, N_DIRECTIONS])
        dims: [action_type, grid_row, grid_col, direction_idx]
        action_type: 0=NOOP  1=PLACE_DRILL  2=PLACE_POLE

    The agent specifies WHERE to place, not WHERE to walk — the env
    auto-navigates the player before every placement.

    Reward:
        primary  — coal drained from drill buffers this step (Lua drain)
        shaped   — +SHAPED_DRILL_ON_COAL_REWARD when a drill is successfully
                   placed on a coal tile (helps escape sparse-reward zone)

    Observation:
        "grid"     — (4, 32, 32) float32
                     ch0: coal resource tiles (fixed)
                     ch1: placed mining drills
                     ch2: placed electric poles
                     ch3: player position
        "features" — (4,) float32
                     [miners_placed, miners_on_coal, poles_placed, coal_total]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance: FactorioInstance,
        task=None,
        max_steps: int = 256,
        scan_radius: int = GRID_CENTER,
        ticks_per_step: int = 60,
        game_speed: int = 10,
    ):
        super().__init__()
        self.instance = instance
        self.task = task
        self.ns = instance.first_namespace
        self.max_steps = max_steps
        self.scan_radius = scan_radius
        self.ticks_per_step = ticks_per_step
        self.game_speed = game_speed

        self._step_count = 0
        self._miners_placed = 0
        self._poles_placed = 0
        self._coal_total = 0.0
        self._coal_patch_bb: Optional[BoundingBox] = None  # set at reset
        self._coal_center = Position(x=0.0, y=0.0)         # set at reset

        self.action_space = gymnasium.spaces.MultiDiscrete(
            [N_ACTION_TYPES, GRID_SIZE, GRID_SIZE, N_DIRECTIONS]
        )
        self.observation_space = gymnasium.spaces.Dict({
            "grid": gymnasium.spaces.Box(
                low=0.0, high=1.0,
                shape=(N_CHANNELS, GRID_SIZE, GRID_SIZE),
                dtype=np.float32,
            ),
            "features": gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(N_FEATURES,),
                dtype=np.float32,
            ),
        })

    # ── Core Gymnasium API ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None) -> Tuple[dict, dict]:
        super().reset(seed=seed)
        if self.task is not None:
            self.task.setup(self.instance)
        else:
            self.instance.reset()

        self._step_count = 0
        self._miners_placed = 0
        self._poles_placed = 0
        self._coal_total = 0.0
        self._coal_patch_bb = None
        self._coal_center = Position(x=0.0, y=0.0)

        # Locate coal patch once — grid stays fixed on it for the whole episode
        self._init_coal_patch()
        return self._get_observation(), {}

    def step(self, action) -> Tuple[dict, float, bool, bool, dict]:
        action_type = int(action[0])
        grid_row    = int(action[1])
        grid_col    = int(action[2])
        dir_idx     = int(action[3])

        shaped, error = self._dispatch_action(action_type, grid_row, grid_col, dir_idx)
        drained = self._advance_game()
        reward = drained + shaped
        self._coal_total += drained
        self._step_count += 1

        truncated = self._step_count >= self.max_steps
        terminated = False
        if truncated and self.task is not None:
            resp = self.task.verify(0.0, self._step_count, self.instance, {})
            terminated = bool(resp.success) if resp is not None else False

        info = {"error": error, "step": self._step_count, "drained": drained}
        return self._get_observation(), reward, terminated, truncated, info

    def close(self):
        try:
            self.instance.cleanup()
        except Exception:
            pass

    # ── Coal patch initialisation ─────────────────────────────────────────────

    def _init_coal_patch(self) -> None:
        try:
            patch = self.ns.get_resource_patch(
                Resource.Coal, Position(x=0.0, y=0.0), self.scan_radius
            )
            if patch is not None:
                self._coal_patch_bb = patch.bounding_box
                bb = patch.bounding_box
                self._coal_center = Position(
                    x=(bb.left_top.x + bb.right_bottom.x) / 2.0,
                    y=(bb.left_top.y + bb.right_bottom.y) / 2.0,
                )
        except Exception:
            pass

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _grid_to_world(self, grid_row: int, grid_col: int) -> Position:
        """Map grid cell to world position, centred on the coal patch."""
        cx = self._coal_center.x
        cy = self._coal_center.y
        return Position(x=cx + (grid_col - GRID_CENTER), y=cy + (grid_row - GRID_CENTER))

    def _world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        cx = self._coal_center.x
        cy = self._coal_center.y
        row = int(round(wy - cy + GRID_CENTER))
        col = int(round(wx - cx + GRID_CENTER))
        return row, col

    def _on_coal(self, pos: Position) -> bool:
        if self._coal_patch_bb is None:
            return False
        bb = self._coal_patch_bb
        return (bb.left_top.x <= pos.x <= bb.right_bottom.x and
                bb.left_top.y <= pos.y <= bb.right_bottom.y)

    # ── Game tick advance + drain ─────────────────────────────────────────────

    def _advance_game(self) -> float:
        """
        Run ticks_per_step game ticks, then drain all drill output buffers.

        Draining ensures drills never stall due to a full buffer, giving a
        sustained reward signal throughout the episode.

        Returns the number of coal units drained this step.
        """
        rc = self.instance.rcon_client
        rc.send_command(f"/sc game.speed = {self.game_speed}")
        rc.send_command("/sc game.tick_paused = false")
        time.sleep(self.ticks_per_step / (60.0 * self.game_speed))
        rc.send_command("/sc game.tick_paused = true")

        drain_cmd = (
            "/sc local d=0; "
            "for _,e in pairs(game.surfaces[1].find_entities_filtered{"
            "type='mining-drill', force='player'}) do "
            "local i=e.get_output_inventory(); "
            "if i then d=d+(i.get_item_count('coal') or 0); i.clear() end "
            "end; rcon.print(d)"
        )
        resp = rc.send_command(drain_cmd)
        try:
            return float(str(resp).strip())
        except Exception:
            return 0.0

    # ── Action dispatch ────────────────────────────────────────────────────────

    def _dispatch_action(
        self, action_type: int, grid_row: int, grid_col: int, dir_idx: int
    ) -> Tuple[float, str]:
        """Execute action. Returns (shaped_reward, error_string)."""
        if action_type == ACTION_NOOP:
            return 0.0, ""

        target = self._grid_to_world(grid_row, grid_col)
        direction = Direction[DIRECTION_MAP[dir_idx]]

        try:
            # Auto-navigate to within reach of the target before placing
            self.ns.move_to(target)

            if action_type == ACTION_PLACE_DRILL:
                self.ns.place_entity(Prototype.ElectricMiningDrill, direction, target)
                self._miners_placed += 1
                shaped = SHAPED_DRILL_ON_COAL_REWARD if self._on_coal(target) else 0.0
                return shaped, ""

            elif action_type == ACTION_PLACE_POLE:
                self.ns.place_entity(Prototype.SmallElectricPole, direction, target)
                self._poles_placed += 1
                return 0.0, ""

        except Exception as exc:
            return 0.0, str(exc)

        return 0.0, ""

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_observation(self) -> dict:
        grid = self._build_grid()
        features = self._build_features(grid)
        return {"grid": grid, "features": features}

    def _build_grid(self) -> np.ndarray:
        grid = np.zeros((N_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        cx = self._coal_center.x
        cy = self._coal_center.y

        # Coal resource (fixed per episode)
        if self._coal_patch_bb is not None:
            bb = self._coal_patch_bb
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    wx = cx + (col - GRID_CENTER)
                    wy = cy + (row - GRID_CENTER)
                    if (bb.left_top.x <= wx <= bb.right_bottom.x and
                            bb.left_top.y <= wy <= bb.right_bottom.y):
                        grid[CH_COAL_RESOURCE, row, col] = 1.0

        # Entities (drills and poles)
        try:
            entities = self.ns.get_entities(
                position=Position(x=cx, y=cy), radius=float(self.scan_radius)
            )
            for ent in entities:
                epos = ent.position
                row, col = self._world_to_grid(epos.x, epos.y)
                if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                    ent_name = getattr(ent, "name", "") or ""
                    if "mining-drill" in ent_name:
                        grid[CH_DRILLS, row, col] = 1.0
                    elif "electric-pole" in ent_name:
                        grid[CH_POLES, row, col] = 1.0
        except Exception:
            pass

        # Player position
        px = self.ns.player_location.x
        py = self.ns.player_location.y
        pr, pc = self._world_to_grid(px, py)
        if 0 <= pr < GRID_SIZE and 0 <= pc < GRID_SIZE:
            grid[CH_PLAYER, pr, pc] = 1.0

        return grid

    def _build_features(self, grid: np.ndarray) -> np.ndarray:
        features = np.zeros(N_FEATURES, dtype=np.float32)
        features[FEAT_MINERS_PLACED] = float(self._miners_placed)
        # Miners on coal = cells where both drill and coal channels are 1
        features[FEAT_MINERS_ON_COAL] = float(
            np.sum((grid[CH_DRILLS] > 0) & (grid[CH_COAL_RESOURCE] > 0))
        )
        features[FEAT_POLES_PLACED] = float(self._poles_placed)
        features[FEAT_COAL_TOTAL] = float(self._coal_total)
        return features
