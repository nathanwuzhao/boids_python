import time
import numpy as np
import pybullet as pb

from arena import Arena
from boid_controller import BoidController
from config import ArenaConfig, BoidParams
from metrics import compute_all_metrics, EpisodeMetrics
from observations import SectorObservation, sector_observations_to_array

arena_cfg = ArenaConfig(num_agents=15, fps=60.0, gui=True)
arena = Arena(num_agents=arena_cfg.num_agents, fps=arena_cfg.fps, gui=arena_cfg.gui)

boid_params = BoidParams(max_speed=2.0, max_force=0.1, 
                         separation_radius=0.7, alignment_radius=0.9, cohesion_radius=1.1,
                         sep_weight=1.5, align_weight=1.0, coh_weight=0.6, target_weight=0.8, 
                         use_kdtree=False, wander_strength=0.0)

controller = BoidController(params=boid_params)

target = np.array([0.0, 0.0], dtype=float)

episode_metrics = EpisodeMetrics()

observer = SectorObservation(
    num_sectors=12, 
    radius=2.0,
    distance_decay=True,
    include_target=True
)

frame_count = 0

while True:
    keys = pb.getKeyboardEvents()

    if ord('r') in keys:
        episode_metrics.reset()
        arena.reset()

    positions, velocities = arena.get_states()

    observations = observer.observe(
        positions=positions,
        velocities=velocities,
        target=target
    )

    obs_array = sector_observations_to_array(observations=observations)

    runtime_override = None
    actions = controller.compute_actions(
        positions=positions, 
        velocities=velocities, 
        target=target, 
        params_override=runtime_override,
    )

    arena.apply_actions(actions)
    arena.step()

    metrics = compute_all_metrics(positions, velocities, target, goal_radius=0.75, safety_radius=0.35)
    episode_metrics.update(metrics)

    if frame_count % arena_cfg.fps == 0:
        agent0_obs = observations[0]

        print({
            "centroid_goal_dist": round(metrics["centroid_goal_dist"], 2),
            "obs_shape": obs_array.shape,
            "agent0_occupancy": np.round(agent0_obs["occupancy"], 2).tolist(),
            "agent0_radial_motion": np.round(agent0_obs["radial_motion"], 2).tolist()

        })

    frame_count += 1
    time.sleep(arena.dt)
    