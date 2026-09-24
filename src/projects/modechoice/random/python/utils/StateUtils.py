def initiate_memory(data, trip_memory):
   
    trip_memory[data.agentID] = {
        "available_modes": data.possibleModeSet,
        "full_data": data,
        "mode_choice_history": trip_memory.get(data.agentID, {}).get("mode_choice_history", []),
        "population": data.subpopulation
    }

def get_chosen_action(agent_id, agent, trip_memory):
    memory = trip_memory.get(agent_id)
    iteration = memory['full_data'].simulationIteration
    
    if not memory: 
        return "pedestrian"
    
    mode = agent.choose_action(agent_id, memory['available_modes'])

    memory["mode"] = mode
    memory["mode_choice_history"].append(mode)
    
    return mode