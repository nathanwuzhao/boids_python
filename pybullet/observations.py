import numpy as np

def normalize_vectors(vectors, eps=1e-8):
    mags = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, mags + eps, out=np.zeros_like(vectors), where=mags > eps)

def angle_wrap(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def heading_from_velocity(velocity, fallback=np.array([1.0, 0.0])):
    speed = np.linalg.norm(velocity)
    if speed < 1e-8:
        return fallback.copy()
    return velocity / speed

def angle_from_vector(v):
    return np.arctan2(v[1], v[0])

class ExactNeighborObservations:
    def __init__(self, radius=1.5):
        self.radius = radius

    def observe(self, positions, velocities, target=None):
        positions = np.asarray(positions, dtype=float)
        velocities = np.asarray(velocities, dtype=float)

        n = positions.shape[0]
        observations = []

        for i in range(n):
            rel_positions = []
            rel_velocities = []
            distances = []

            for j in range(n):
                if i == j:
                    continue

                rel_pos = positions[j] - positions[i]
                dist = np.linalg.norm(rel_pos)

                if dist <= self.radius:
                    rel_positions.append(rel_pos)
                    rel_velocities.append(velocities[j] - velocities[i])
                    distances.append(dist)

            obs = {
                "agent_index": i,
                "position": positions[i].copy(),
                "velocity": velocities[i].copy(),
                "neighbor_rel_positions": np.array(rel_positions, dtype=float),
                "neighbor_rel_velocities": np.array(rel_velocities, dtype=float),
                "neighbor_distances": np.array(distances, dtype=float)
            }

            if target is not None:
                obs["target_vector"] = np.asarray(target, dtype=float) - positions[i]
            
            observations.append(obs)

        return observations