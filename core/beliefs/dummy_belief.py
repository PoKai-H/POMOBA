import numpy as np


class DummyBelief:
    """Fixed 4-way strategy belief for early rollout wiring."""

    def __init__(self, num_strategies=4):
        self.num_strategies = num_strategies
        self.last_obs = None
        self.last_belief = np.full(
            self.num_strategies,
            1.0 / self.num_strategies,
            dtype=np.float32,
        )

    def update(self, obs_vec):
        self.last_obs = obs_vec
        return self.last_belief.copy()
