def prepare_state(data, trip_memory):
    agent_id = data.agentID
    feature_map = data.features

    agent_trip_memory = trip_memory.get(data.agentID)

    new_state = tuple(feature_map["rawBitStateRepresentation"])

    if agent_trip_memory:
        previous_state = agent_trip_memory.get("state")
        previous_action = agent_trip_memory.get("mode")
    else:
        previous_state = None
        previous_action = None
    
    trip_memory[agent_id] = {
        "state": new_state,                             # Current state (S_t)
        "previous_state": previous_state,               # Older state (S_{t-1})
        "previous_mode": previous_action,               # Older action (A_{t-1})
        "available_modes": data.possibleModeSet,
        "full_data": data,
        "action_history": agent_trip_memory.get("action_history", []) if agent_trip_memory else [],
        "reward_history": agent_trip_memory.get("reward_history", []) if agent_trip_memory else [],
        "population": data.subpopulation
    }

    return new_state

def get_chosen_action(agent_id, agent, trip_memory):
    memory = trip_memory.get(agent_id)
    iteration = memory['full_data'].simulationIteration
    
    if not memory: 
        return "pedestrian"
    
    # Update the epsilon value for the iteration
    agent.decay_epsilon(iteration)
    mode = agent.choose_action(agent_id, memory['state'], memory['available_modes'])

    memory["mode"] = mode
    memory["action_history"].append(mode)
    
    return mode
