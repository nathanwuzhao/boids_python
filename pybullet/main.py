import time
import numpy as np
import pybullet as pb

from arena import Arena
from boid_controller import BoidController

fps = 60
arena = Arena(num_agents=15, fps=fps, gui=True)
controller = BoidController(
    max_speed=2.0,
    max_force=0.1,
    separation_radius=0.7, 
    alignment_radius=0.9,
    cohesion_radius=1.1,
    sep_weight=1.5,
    align_weight=1.0,
    coh_weight=0.6,
    target_weight=0.8,
    use_kdtree=False,
    wander_strength=0.0
)

target = np.array([0.0, 0.0], dtype=float)

while True:
    keys = pb.getKeyboardEvents()
    if ord('r') in keys:
        arena.reset()

    pos, vel = arena.get_states()

    actions = controller.compute_actions(pos, vel, target=target)
    
    arena.apply_actions(actions)
    arena.step()
    
    time.sleep(arena.dt)