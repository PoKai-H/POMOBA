import numpy as np


class DummyBelief:
    """Fixed 3-way strategy belief for early rollout wiring."""

    def __init__(self, num_strategies=3):
        self.num_strategies = num_strategies
        self.last_obs = None
        self.last_belief = np.full(
            self.num_strategies,
            1.0 / self.num_strategies,
            dtype=np.float32,
        )

    def update(self, obs):
        self.last_obs = obs
        return self.last_belief.copy()
