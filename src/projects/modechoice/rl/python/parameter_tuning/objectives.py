import numpy as np

def policy_evolution_rate(reward_history, window_size=50, threshold=0.02):
    reward_moving_averages = []

    # 1. Generate smoothed reward curve
    for i in range(len(reward_history) - window_size + 1):
        window_slice = reward_history[i : i + window_size]
        reward_moving_averages.append(sum(window_slice) / window_size)
        
    if len(reward_moving_averages) < 2:
        return float(len(reward_history) - 1), reward_moving_averages

    saturation_iteration = float(len(reward_history) - 1)

    # 2. Find the absolute highest peak that this moving average ever reaches
    global_max = max(reward_moving_averages)
    
    allowed_distance = abs(global_max) * threshold

    for idx, current_avg in enumerate(reward_moving_averages):
        
        if (global_max - current_avg) < allowed_distance:
            saturation_iteration = float(idx + window_size)
            break
            
    return saturation_iteration, reward_moving_averages

def settled_policy_metrics(policy_saturation_iteration, total_iterations, delta_q_history, reward_history, evaluation_window_size=20):

    if policy_saturation_iteration < total_iterations:
        start_window = int(policy_saturation_iteration) - 1
        end_window = start_window + evaluation_window_size
    else:
        start_window = max(0, total_iterations - evaluation_window_size)
        end_window = total_iterations
        
    converged_rewards = reward_history[start_window:end_window]

    if len(converged_rewards) == 0:
        converged_rewards = reward_history[-max(1, evaluation_window_size):]
    
    reward_at_convergence = np.mean(converged_rewards)

    reward_mean = np.mean(reward_history[start_window:])
    std_deviation = np.std(reward_history[start_window:])
    
    if reward_at_convergence != 0:
        relative_policy_variance = float(std_deviation / abs(reward_mean))
    else:
        relative_policy_variance = float(std_deviation)

    if relative_policy_variance > 0.02:
        reward_at_convergence -= 300 * relative_policy_variance

    delta_q_at_stabilization = np.mean(delta_q_history[start_window:end_window]) if delta_q_history else 0.0
    
    #variance_distance_from_target = max(0.0, relative_policy_variance - 0.02)
    
    return reward_at_convergence, delta_q_at_stabilization, relative_policy_variance 