import random

from numpy import long

class RandomModeAgent:
    
    _random_seed = 42

    @classmethod
    def configure_global_parameters(cls, config_args: dict):
        """
        Dynamically configures class-level shared parameters from any arbitrary dictionary 
        passed by MATSim/Java, without needing to know every key ahead of time.
        """
        if config_args is not {}:
            random_seed = config_args.get("randomSeed")
            random.seed(random_seed)
        else:
            random.seed(cls.RANDOM_SEED)

    def __init__(self, unique_id, args={}):
        """Initializes a unique instance for a specific ID."""
        self._id = unique_id

        for key, value in args.items():
            setattr(self, key, value)

    @property
    def seed(self):
        return self._random_seed
    
    def choose_action(self, available_modes):
        """Selects a random mode from the available modes list."""
        # Ensure available modes is valid and not empty
        if not available_modes:
            return "pedestrian"  # fallback default if empty

        return random.choice(available_modes)