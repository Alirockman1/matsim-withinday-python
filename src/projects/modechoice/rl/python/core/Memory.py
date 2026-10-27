class Memory:
    """Tracks an agent's state and state-action occurrences."""

    def __init__(self):
        self._states = {}

    def record_state(self, state, available_actions):
        """Record one visit to a state."""
        if state not in self._states:
            self._states[state] = {
                "state_count": 0,
                "action_counts": {
                    action: 0 for action in available_actions
                },
            }

        entry = self._states[state]
        entry["state_count"] += 1

        # Include newly available actions without resetting existing counts
        for action in available_actions:
            entry["action_counts"].setdefault(action, 0)

    def record_action(self, state, action):
        """Record an action actually selected in a state."""
        if state not in self._states:
            self._states[state] = {
                "state_count": 0,
                "action_counts": {},
            }

        counts = self._states[state]["action_counts"]
        counts[action] = counts.get(action, 0) + 1

    def get_state_count(self, state):
        """Return the number of times a state has been visited."""
        entry = self._states.get(state)
        return entry["state_count"] if entry else 0

    def get_action_count(self, state, action):
        """Return how often an action was selected in a state."""
        entry = self._states.get(state)
        return entry["action_counts"].get(action, 0) if entry else 0

    def summary(self):
        """Returns aggregate metrics counts from memory."""
        total_states_visited = sum(entry.get("state_count", 0) for entry in self._states.values())
        unique_states_visited = len(self._states)
        total_state_action_combos = sum(
            sum(entry.get("action_counts", {}).values()) 
            for entry in self._states.values()
        )
        return {
            "total_states_visited": total_states_visited,
            "unique_states_visited": unique_states_visited,
            "total_state_action_combinations": total_state_action_combos,
        }

    def details(self):
        """Returns the dictionary mapping unique states to occurrences and action breakdown."""
        return {
            str(state): {
                "state_count": entry.get("state_count", 0),
                "action_counts": entry.get("action_counts", {}),
            }
            for state, entry in self._states.items()
        }

    def get_data(self):
        """Return all stored exploration statistics for persistence."""
        return self._states

    def load_data(self, data):
        """Restore exploration statistics from saved data."""
        self._states = data if isinstance(data, dict) else {}