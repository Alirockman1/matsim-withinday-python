def update_experience(data, agent, daily_stats):
    feature_map = data.features
    
    # Accumulate Daily Stats
    stats = daily_stats.setdefault(data.agentID, {"reward": 0.0, "matsim_score": 0.0})
    stats["reward"] += data.reward
    stats["matsim_score"] += data.matsimScore

    # Determine the next state (Terminal state is None)
    next_state = None

    if not data.isTerminal:
        next_state = tuple(feature_map["nextRawBitStateRepresentation"])
        agent.init_state(data.agentID, next_state)

    return data.reward, next_state, data.isTerminal


def finalize_session(data, trip_memory, agent, logger):
    
    agent_id = data.agentID
    end_of_day_score = data.accumulativeScore
    end_of_day_reward = data.accumulativeReward

    memory = trip_memory.pop(agent_id)
    
    logger.log_day_summary(
        agent_id=agent_id,
        total_reward=end_of_day_reward,
        iteration=memory['full_data'].simulationIteration,
        q_table=agent.q_table
    )

    """ logger.save_to_csv(
        iteration=memory['full_data'].simulationIteration,
        epsilon=agent.epsilon,
        agent_id=agent_id,
        mode_history=memory['mode_choice_history'],
        matsim_score=end_of_day_score,
        total_reward=end_of_day_reward
    )"""