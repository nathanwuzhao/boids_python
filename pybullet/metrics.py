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

