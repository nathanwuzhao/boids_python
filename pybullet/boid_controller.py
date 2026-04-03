import numpy as np
from scipy.spatial import cKDTree

def limit_magnitude(vectors, max_magnitude):
    mags = np.linalg.norm(vectors, axis=1, keepdims=True)
    scale = np.ones_like(mags)
    mask = mags > max_magnitude
    scale[mask] = max_magnitude / (mags[mask] + 1e-8)
    return vectors * scale

class BoidController:
    def __init__(
        self,
        max_speed=2.0,
        max_force=0.08,
        separation_radius=0.5, 
        alignment_radius=0.8,
        cohesion_radius=1.0,
        sep_weight=1.2,
        align_weight=1.0,
        coh_weight=0.6,
        target_weight=0.8,
        use_kdtree=False,
        wander_strength=0.0,
    ):
        self.max_speed = max_speed
        self.max_force = max_force

        self.separation_radius = separation_radius
        self.alignment_radius = alignment_radius
        self.cohesion_radius = cohesion_radius
        
        self.sep_weight = sep_weight
        self.align_weight = align_weight
        self.coh_weight = coh_weight
        self.target_weight = target_weight

        self.use_kdtree = use_kdtree
        self.wander_strength = wander_strength

    def steer_toward(self, current_velocities, desired_vectors):
        desired_mags = np.linalg.norm(desired_vectors, axis=1, keepdims=True)
        desired_dirs = np.divide(desired_vectors, desired_mags + 1e-8, 
                                 out=np.zeros_like(desired_vectors), where=desired_mags > 1e-8)
        
        desired_velocities = desired_dirs * self.max_speed
        steering = desired_velocities - current_velocities
        return limit_magnitude(steering, self.max_force)
    
    def neighbor_lists(self, positions):
        if not self.use_kdtree:
            return None
        
        max_radius = max(self.separation_radius, self.alignment_radius, self.cohesion_radius)
        tree = cKDTree(positions)
        return tree.query_ball_point(positions, r=max_radius)
    
    def compute_actions(self, positions, velocities, target=None):
        n = positions.shape[0]

        sep_sum = np.zeros((n, 2), dtype=float)
        align_sum = np.zeros((n, 2), dtype=float)
        coh_sum = np.zeros((n, 2), dtype=float)

        sep_count = np.zeros(n, dtype=int)
        align_count = np.zeros(n, dtype=int)
        coh_count = np.zeros(n, dtype=int)

        sep_r2 = self.separation_radius ** 2
        align_r2 = self.alignment_radius ** 2
        coh_r2 = self.cohesion_radius ** 2

        neighbor_lists = self.neighbor_lists(positions)

        for i in range(n):
            if neighbor_lists is None:
                candidates = range(n)
            else:
                candidates = neighbor_lists[i]

            for j in candidates:
                if i == j:
                    continue

                diff = positions[i] - positions[j]
                d2 = diff[0] * diff[0] + diff[1] * diff[1]

                if d2 < 1e-12:
                    continue

                if d2 < sep_r2:
                    sep_sum[i] += diff / (d2 + 1e-6)
                    sep_count[i] += 1

                if d2 < align_r2:
                    align_sum[i] += velocities[j]
                    align_count += 1

                if d2 < coh_r2:
                    coh_sum[i] += positions[j]
                    coh_count[i] += 1

        steering_total = np.zeros((n, 2), dtype=float)

        #separation
        sep_mask = sep_count > 0
        if np.any(sep_mask):
            sep_desired = np.zeros((n, 2), dtype=float)
            sep_desired[sep_mask] = sep_sum[sep_mask] / sep_count[sep_mask, None]
            sep_steer = self.steer_toward(velocities, sep_desired)
            steering_total += sep_steer * self.sep_weight

        #alignment
        align_mask = align_count > 0
        if np.any(align_mask):
            align_desired = np.zeros((n, 2), dtype=float)
            align_desired[align_mask] = align_sum[align_mask] / align_count[align_mask, None]
            align_steer = self.steer_toward(velocities, align_desired)
            steering_total += align_steer * self.align_weight

        #cohesion
        coh_mask = coh_count > 0
        if np.any(coh_mask):
            coh_centers = np.zeros((n, 2), dtype=float)
            coh_centers[coh_mask] = coh_sum[coh_mask] / coh_count[coh_mask, None]
            coh_desired = coh_centers - positions
            coh_steer = self.steer_toward(velocities, coh_desired)
            steering_total += coh_steer * self.coh_weight

        # Target seeking
        if target is not None:
            target = np.asarray(target, dtype=float)
            target_vectors = target[None, :] - positions
            target_steer = self.steer_toward(velocities, target_vectors)
            steering_total += target_steer * self.target_weight

        if self.wander_strength > 0:
            wander = np.random.uniform(-1.0, 1.0, size=(n, 2))
            wander = limit_magnitude(wander, 1.0)
            steering_total += wander * self.wander_strength

        desired_velocities = velocities + steering_total
        desired_velocities = limit_magnitude(desired_velocities, self.max_speed)

        return desired_velocities