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
    
class SectorObservation:
    def __init__(self, num_sectors=16, radius=2.0, distance_decay=True, include_target=True):
        self.num_sectors = num_sectors
        self.radius = radius
        self.distance_decay = distance_decay
        self.include_target = include_target

    def observe(self, positions, velocities, target=None):
        positions = np.asarray(positions, dtype=float)
        velocities = np.asarray(velocities, dtype=float)

        n = positions.shape[0]
        observations = []

        for i in range(n):
            heading = heading_from_velocity(velocities[i])
            heading_angle = angle_from_vector(heading)

            occupancy = np.zeros(self.num_sectors, dtype=float)
            closeness = np.zeros(self.num_sectors, dtype=float)
            radial_motion = np.zeros(self.num_sectors, dtype=float)
            tangential_motion = np.zeros(self.num_sectors, dtype=float)

            for j in range(n):
                if i == j:
                    continue

                rel_pos = positions[j] - positions[i]
                dist = np.linalg.norm(rel_pos)

                if dist < 1e-8 or dist > self.radius:
                    continue

                rel_vel = velocities[j] - velocities[i]

                world_angle = angle_from_vector(rel_pos)
                local_angle = angle_wrap(world_angle - heading_angle)

                sector = self.angle_to_sector(local_angle)

                rel_dir = rel_pos / dist
                tangent_dir = np.array([-rel_dir[1], rel_dir[0]])

                radial = np.dot(rel_vel, rel_dir)
                tangential = np.dot(rel_vel, tangent_dir)

                if self.distance_decay:
                    strength = 1.0 - (dist / self.radius)
                else:
                    strength = 1.0

                occupancy[sector] += strength
                closeness[sector] = max(closeness[sector], strength)
                radial_motion[sector] += radial * strength
                tangential_motion[sector] += tangential * strength  

            occupancy = np.clip(occupancy, 0.0, 1.0)

            obs = {
                "agent_index": i,
                "heading": heading,
                "occupancy": occupancy, 
                "closeness": closeness,
                "radial_motion": radial_motion, 
                "tangential_motion": tangential_motion
            }

            if self.include_target and target is not None:
                target_vec = np.asarray(target, dtype=float) - positions[i]
                target_dist = np.linalg.norm(target_vec)

                if target_dist > 1e-8:
                    target_angle = angle_wrap(angle_from_vector(target_vec) - heading_angle)
                    target_sector = self.angle_to_sector(target_angle)
                    target_dir_local = np.array([np.cos(target_angle), np.sin(target_angle)])
                else:
                    target_angle = 0.0
                    target_sector = 0
                    target_dir_local = np.zeros(2)

                obs["target_dist"] = float(target_dist)
                obs["target_angle"] = float(target_angle)
                obs["target_sector"] = int(target_sector)
                obs["target_dir_local"] = target_dir_local

            observations.append(obs)

        return observations

    def angle_to_sector(self, angle):
        normalized = (angle + np.pi) / (2 * np.pi)
        sector = int(normalized * self.num_sectors)
        return min(max(sector, 0), self.num_sectors - 1)

def sector_observations_to_array(observations):
    rows = []

    for obs in observations:
        row_parts =[
            obs["occupancy"],
            obs["closeness"],
            obs["radial_motion"],
            obs["tangential_motion"]
        ]

        if "target_dist" in obs:
            row_parts.append(np.array([obs["target_dist"], obs["target_angle"]], dtype=float))

        rows.append(np.concatenate(row_parts))

    return np.array(rows, dtype=float)