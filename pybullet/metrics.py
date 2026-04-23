import numpy as np

def compute_centroid(positions):
    return np.mean(positions, axis=0)

def compute_swarm_spread(positions):
    centroid = compute_centroid(positions)
    dists = np.linalg.norm(positions - centroid, axis=1)

    return {
        "mean_dist_to_centroid": float(np.mean(dists)),
        "max_dist_to_centroid": float(np.max(dists))
    }

def compute_goal_metrics(positions, target, goal_radius=0.5):
    target = np.asarray(target, dtype=float)
    centroid = compute_centroid(positions)

    agent_goal_dists = np.linalg.norm(positions - target, axis=1)
    centroid_goal_dist = float(np.linalg.norm(centroid - target))
    mean_agent_goal_dist = float(np.mean(agent_goal_dists))
    fraction_at_goal = float(np.mean(agent_goal_dists <= goal_radius))

    return {
        "centroid_goal_dist": centroid_goal_dist,
        "mean_agent_goal_dist": mean_agent_goal_dist,
        "fraction_at_goal": fraction_at_goal
    }

def compute_alignment_metrics(velocities):
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)

    directions = np.divide(
        velocities, 
        speeds + 1e-8, 
        out = np.zeros_like(velocities),
        where=speeds > 1e-8,
    )

    mean_direction = np.mean(directions, axis=0)
    alignment = float(np.linalg.norm(mean_direction))

    return {
        "alignment": alignment
    }

def compute_min_interagent_distance(positions):
    n = positions.shape[0]
    if n < 2:
        return { "min_interagent_dist": float("inf")}
    
    min_dist = float("inf")

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                min_dist = d

    return {
        "min_interagent_dist": float(min_dist)
    }

def compute_collision_risk_metric(positions, safety_radius):
    n = positions.shape[0]
    if n < 2: 
        return {
            "close_pair_fraction": 0.0,
            "num_close_pairs": 0,
            "num_pairs": 0
        }
    
    num_pairs = 0
    num_close_pairs = 0
    r2 = safety_radius ** 2

    for i in range(n):
        for j in range(i + 1, n):
            num_pairs += 1
            diff = positions[i] - positions[j]
            d2 = diff[0] * diff[0] + diff[1] * diff[1]
            if d2 < r2:
                num_close_pairs += 1

    close_pair_fraction = num_close_pairs / num_pairs if num_pairs > 0 else 0.0

    return {
        "close_pair_fraction": float(close_pair_fraction),
        "num_close_pairs": int(num_close_pairs),
        "num_pairs": int(num_pairs)
    }

def compute_all_metrics(positions, velocities, target=None, goal_radius=0.5, safety_radius=0.3):
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)

    metrics = {}

    centroid = compute_centroid(positions=positions)
    metrics["centroid_x"] = float(centroid[0])
    metrics["centroid_y"] = float(centroid[1])

    metrics.update(compute_swarm_spread(positions=positions))
    metrics.update(compute_alignment_metrics(velocities=velocities))
    metrics.update(compute_min_interagent_distance(positions=positions))
    metrics.update(compute_collision_risk_metric(positions=positions, safety_radius=safety_radius))

    mean_speed =float(np.mean(np.linalg.norm(velocities, axis=1)))
    metrics["mean_speed"] = mean_speed

    if target is not None:
        metrics.update(compute_goal_metrics(positions, target, goal_radius))

    return metrics