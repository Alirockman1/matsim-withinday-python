def update_reward(data, agent_trip_memory):
    reward   = data.reward
    matsim_score = data.matsimScore
    print(f"The reward being updated is {reward}")
    agent_trip_memory["reward_history"].append(reward)


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