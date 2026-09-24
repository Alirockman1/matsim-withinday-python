def prepare_state(data, trip_memory):
    #state = (data.encodedStateString)
    feature_map = data.features
    state = tuple(feature_map["rawBitStateRepresentation"])
    
    trip_memory[data.agentID] = {
        "state": state,
        "available_modes": data.possibleModeSet,
        "full_data": data,
        "mode_choice_history": trip_memory.get(data.agentID, {}).get("mode_choice_history", []),
        "population": data.subpopulation
    }

    return state

def get_chosen_action(agent_id, agent, trip_memory):
    memory = trip_memory.get(agent_id)
    iteration = memory['full_data'].simulationIteration
    
    if not memory: 
        return "pedestrian"
    
    # Update the epsilon value for the iteration
    agent.decay_epsilon(iteration)
    mode = agent.choose_action(agent_id, memory['state'], memory['available_modes'])

    memory["mode"] = mode
    memory["mode_choice_history"].append(mode)
    
    return mode
