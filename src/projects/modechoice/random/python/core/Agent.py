import random

class RandomModeAgent:

    def __init__(self, random_seed):
        self._random_seed = random_seed
        random.seed(self._random_seed)

    @property
    def seed(self):
        return self._random_seed
    
    def choose_action(self, agent_id, available_modes):
        """Selects a random mode from the available modes list."""
        # Ensure available modes is valid and not empty
        if not available_modes:
            return "car"  # fallback default if empty

        action = random.choice(available_modes)

        return action