import numpy as np


class DummyEnv:
    def __init__(self):
        self.t = 0

    def reset(self, config):
        self.t = 0
        return self._get_obs()

    def step(self, action):
        self.t += 1

        obs = self._get_obs()
        reward = np.random.randn()

        done = self.t > 50

        return obs, reward, done, {}

    def _get_obs(self):
        return {
            "timestep": self.t,
            "self": {
                "id": 0,
                "team": "ally",
                "hp": np.random.rand(),
                "position": [np.random.randn(), np.random.randn()],
                "status": {
                    "alive": True
                },
            },
            "agents": [
                {
                    "id": 1,
                    "team": "enemy",
                    "visible": True,
                    "hp": np.random.rand(),
                    "relative_position": [np.random.randn(), np.random.randn()],
                    "status": {
                        "alive": True
                    },
                }
            ],
            "objects": [],
            "extensions": {},
        }
