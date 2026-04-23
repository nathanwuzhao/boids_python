import numpy as np
from dataclasses import replace
from scipy.spatial import cKDTree

from config import BoidParams

def limit_magnitude(vectors, max_magnitude):
    mags = np.linalg.norm(vectors, axis=1, keepdims=True)
    scale = np.ones_like(mags)
    mask = mags > max_magnitude
    scale[mask] = max_magnitude / (mags[mask] + 1e-8)
    return vectors * scale

class BoidController:
    def __init__(self, params: BoidParams):
        self.base_params = params

    def resolve_params(self, params_override=None):
        if params_override is None:
            return self.base_params
        
        if isinstance(params_override, BoidParams):
            return params_override
        
        if isinstance(params_override, dict):
            return replace(self.base_params, **params_override)
        
        raise TypeError("params_override must be None, dict, or BoidParams")
    
    def steer_toward(self, current_velocities, desired_vectors, params: BoidParams):
        desired_mags = np.linalg.norm(desired_vectors, axis=1, keepdims=True)

        desired_dirs = np.divide(
            desired_vectors,
            desired_mags + 1e-8,
            out=np.zeros_like(desired_vectors),
            where=desired_mags > 1e-8,
        )

        desired_velocities = desired_dirs * params.max_speed
        steering = desired_velocities - current_velocities

        return limit_magnitude(steering, params.max_force)

    def neighbor_lists(self, positions, params: BoidParams):
        if not params.use_kdtree:
            return None

        max_radius = max(
            params.separation_radius,
            params.alignment_radius,
            params.cohesion_radius,
        )

        tree = cKDTree(positions)
        return tree.query_ball_point(positions, r=max_radius)

    def compute_actions(self, positions, velocities, target=None, params_override=None):
        params = self.resolve_params(params_override)

        n = positions.shape[0]

        sep_sum = np.zeros((n, 2), dtype=float)
        align_sum = np.zeros((n, 2), dtype=float)
        coh_sum = np.zeros((n, 2), dtype=float)

        sep_count = np.zeros(n, dtype=int)
        align_count = np.zeros(n, dtype=int)
        coh_count = np.zeros(n, dtype=int)

        sep_r2 = params.separation_radius ** 2
        align_r2 = params.alignment_radius ** 2
        coh_r2 = params.cohesion_radius ** 2

        neighbor_lists = self.neighbor_lists(positions, params)

        for i in range(n):
            candidates = range(n) if neighbor_lists is None else neighbor_lists[i]

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
                    align_count[i] += 1

                if d2 < coh_r2:
                    coh_sum[i] += positions[j]
                    coh_count[i] += 1

        steering_total = np.zeros((n, 2), dtype=float)

        # Separation
        sep_mask = sep_count > 0
        if np.any(sep_mask):
            sep_desired = np.zeros((n, 2), dtype=float)
            sep_desired[sep_mask] = sep_sum[sep_mask] / sep_count[sep_mask, None]
            sep_steer = self.steer_toward(velocities, sep_desired, params)
            steering_total += sep_steer * params.sep_weight

        # Alignment
        align_mask = align_count > 0
        if np.any(align_mask):
            align_desired = np.zeros((n, 2), dtype=float)
            align_desired[align_mask] = align_sum[align_mask] / align_count[align_mask, None]
            align_steer = self.steer_toward(velocities, align_desired, params)
            steering_total += align_steer * params.align_weight

        # Cohesion
        coh_mask = coh_count > 0
        if np.any(coh_mask):
            coh_centers = np.zeros((n, 2), dtype=float)
            coh_centers[coh_mask] = coh_sum[coh_mask] / coh_count[coh_mask, None]
            coh_desired = coh_centers - positions
            coh_steer = self.steer_toward(velocities, coh_desired, params)
            steering_total += coh_steer * params.coh_weight

        # Target seeking
        if target is not None:
            target = np.asarray(target, dtype=float)
            target_vectors = target[None, :] - positions
            target_steer = self.steer_toward(velocities, target_vectors, params)
            steering_total += target_steer * params.target_weight

        # Optional wander
        if params.wander_strength > 0:
            wander = np.random.uniform(-1.0, 1.0, size=(n, 2))
            wander = limit_magnitude(wander, 1.0)
            steering_total += wander * params.wander_strength

        desired_velocities = velocities + steering_total
        desired_velocities = limit_magnitude(desired_velocities, params.max_speed)

        return desired_velocities