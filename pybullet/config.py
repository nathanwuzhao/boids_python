from dataclasses import dataclass

@dataclass
class ArenaConfig:
    num_agents: int = 15
    fps: float = 60.0
    gui: bool = True

@dataclass
class BoidParams:
    max_speed: float = 2.0
    max_force: float = 0.15

    separation_radius: float = 1.0
    alignment_radius: float = 1.1
    cohesion_radius: float = 1.2

    sep_weight: float = 1.5
    align_weight: float = 1.0
    coh_weight: float = 0.6
    target_weight: float = 0.5
    
    use_kdtree: bool = False
    wander_strength: float = 0.0