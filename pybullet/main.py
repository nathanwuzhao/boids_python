import numpy as np
from arena import Arena
import time

fps = 60
arena = Arena(num_agents=10, fps=fps, gui=True)

while True:
    pos, vel = arena.get_states()

    actions = np.random.uniform(-1, 1, size=(len(pos), 2))
    arena.apply_actions(actions)
    arena.step()
    
    time.sleep(1/fps)