from __future__ import annotations

from typing import Dict, Optional

from fle.env import FactorioInstance
from fle.agents import TaskResponse
from fle.commons.models.achievements import ProductionFlows
from fle.eval.tasks.task_abc import TaskABC


class CoalMiningTask(TaskABC):
    """
    Task: place electric mining drills on coal and connect them to the power grid.
    Success when coal production reaches >= COAL_RATE_THRESHOLD units/second.

    Setup places a Factorio electric-energy-interface (vanilla entity, ~10 MW)
    at (0, 5) near spawn so any placed electric poles/drills get power automatically.
    """

    COAL_MINING_TASK_KEY = "coal_mining_rl"
    STARTING_INVENTORY: Dict[str, int] = {
        "electric-mining-drill": 5,
        "small-electric-pole": 10,
    }
    COAL_RATE_THRESHOLD = 1.0  # coal per second

    def __init__(self, trajectory_length: int = 256):
        super().__init__(
            trajectory_length=trajectory_length,
            starting_inventory=self.STARTING_INVENTORY,
            goal_description=(
                "Place electric mining drills on the coal patch and connect them "
                "to a power pole. Achieve at least 1 coal per second production rate."
            ),
            task_key=self.COAL_MINING_TASK_KEY,
            all_technology_reserached=True,
        )

    def setup_instance(self, instance: FactorioInstance) -> None:
        # Place the power source (EEI) directly at the coal patch centre so
        # any pole placed on the coal patch connects immediately to the grid.
        # One anchor pole is pre-placed next to the EEI so the agent only needs
        # to extend the network, not bootstrap it from scratch.
        setup_lua = (
            "/sc "
            "local s = game.surfaces[1]; "
            "local coal = s.find_entities_filtered{type='resource', name='coal', position={0,0}, radius=64}; "
            "local cx, cy = 0, 0; "
            "if #coal > 0 then cx = coal[1].position.x; cy = coal[1].position.y end; "
            "local eei = s.create_entity{name='electric-energy-interface', position={cx, cy}, force='player', create_build_effect_smoke=false}; "
            "if eei then eei.power_production = 50000000; eei.energy = 50000000 end; "
            # Anchor pole adjacent to EEI so the electric network starts here
            "s.create_entity{name='small-electric-pole', position={cx+2, cy}, force='player', create_build_effect_smoke=false}"
        )
        instance.rcon_client.send_command(setup_lua)

    def verify(
        self,
        score: float,
        step: int,
        instance: FactorioInstance,
        step_statistics: dict,
    ) -> TaskResponse:
        ns = instance.first_namespace
        try:
            flows = ProductionFlows.from_dict(ns._get_production_stats())
            # output["coal"] is cumulative; use step_statistics delta if provided,
            # otherwise fall back to total produced vs game ticks elapsed.
            coal_total = float(flows.output.get("coal", 0.0))
            elapsed_ticks = max(1, instance.get_elapsed_ticks())
            coal_rate = coal_total / elapsed_ticks * 60  # ticks→seconds (60 ticks/s)
        except Exception:
            coal_rate = 0.0

        success = coal_rate >= self.COAL_RATE_THRESHOLD
        return TaskResponse(success=success, meta={"coal_rate": coal_rate})
