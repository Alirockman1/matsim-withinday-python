def initiate_memory(agent_tag, data, trip_memory):
    trip_memory[agent_tag] = {
        "available_modes": data.possibleModeSet,
        "full_data": data,
        "mode_choice_history": trip_memory.get(agent_tag, {}).get("mode_choice_history", []),
        "population": data.subpopulation
    }

def get_chosen_action(agent_tag, agent, trip_memory):
    memory = trip_memory.get(agent_tag)
    iteration = memory['full_data'].simulationIteration
    
    if not memory: 
        return "pedestrian"
    
    mode = agent.choose_action(memory['available_modes'])

    memory["mode"] = mode
    memory["mode_choice_history"].append(mode)
    
    return mode