import time
import numpy as np
import pybullet as pb

from arena import Arena
from boid_controller import BoidController

fps = 60
arena = Arena(num_agents=10, fps=fps, gui=True)
controller = BoidController(
    max_speed=2.0,
    max_force=0.08,
    separation_radius=0.5, 
    alignment_radius=0.8,
    cohesion_radius=1.0,
    sep_weight=1.2,
    align_weight=1.0,
    coh_weight=0.6,
    target_weight=0.8,
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
    
    time.sleep(1/fps)