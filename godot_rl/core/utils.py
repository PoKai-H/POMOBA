import importlib
import re

import gymnasium as gym
import numpy as np


def lod_to_dol(lod):
    return {k: [dic[k] for dic in lod] for k in lod[0]}


def dol_to_lod(dol):
    return [dict(zip(dol, t)) for t in zip(*dol.values())]


def convert_macos_path(env_path):
    """
    Convert `Demo.app` into the actual executable path inside the app bundle.
    """
    filenames = re.findall(r"[^\/]+(?=\.)", env_path)
    assert len(filenames) == 1, "Could not convert the Godot app path on macOS."
    return env_path + "/Contents/MacOS/" + filenames[0]


class ActionSpaceProcessor:
    """
    Minimal action-space adapter used by `GodotEnv`.

    The local training code keeps `convert_action_space=False`, so most of this
    class exists for compatibility with the original package API.
    """

    def __init__(self, action_space: gym.spaces.Tuple, convert) -> None:
        self._original_action_space = action_space
        self._convert = convert
        self._all_actions_discrete = all(
            isinstance(space, gym.spaces.Discrete) for space in action_space.spaces
        )
        self._only_one_action_space = len(action_space) == 1

        if self._only_one_action_space and self._all_actions_discrete:
            self.converted_action_space = action_space[0]
            return

        space_size = 0
        if convert:
            use_multi_discrete_spaces = False
            multi_discrete_spaces = np.array([])
            if isinstance(action_space, gym.spaces.Tuple):
                if self._all_actions_discrete:
                    use_multi_discrete_spaces = True
                    for space in action_space.spaces:
                        multi_discrete_spaces = np.append(multi_discrete_spaces, space.n)
                else:
                    for space in action_space.spaces:
                        if isinstance(space, gym.spaces.Box):
                            assert len(space.shape) == 1
                            space_size += space.shape[0]
                        elif isinstance(space, gym.spaces.Discrete):
                            if space.n > 2:
                                raise NotImplementedError(
                                    "Mixed continuous/discrete actions only support "
                                    "binary discrete branches."
                                )
                            space_size += 1
                        else:
                            raise NotImplementedError
            elif isinstance(action_space, gym.spaces.Dict):
                raise NotImplementedError
            else:
                assert isinstance(action_space, (gym.spaces.Box, gym.spaces.Discrete))
                return

            if use_multi_discrete_spaces:
                self.converted_action_space = gym.spaces.MultiDiscrete(multi_discrete_spaces)
            else:
                self.converted_action_space = gym.spaces.Box(-1, 1, shape=[space_size])

    @property
    def action_space(self):
        if not self._convert:
            return self._original_action_space
        return self.converted_action_space

    def to_original_dist(self, action):
        if not self._convert:
            return action

        if self._only_one_action_space and self._all_actions_discrete:
            return [action]

        original_action = []
        counter = 0
        integer_actions = action.dtype == np.int64

        for space in self._original_action_space.spaces:
            if isinstance(space, gym.spaces.Box):
                assert len(space.shape) == 1
                original_action.append(action[:, counter : counter + space.shape[0]])
                counter += space.shape[0]
            elif isinstance(space, gym.spaces.Discrete):
                if integer_actions:
                    discrete_actions = action[:, counter]
                else:
                    if space.n > 2:
                        raise NotImplementedError(
                            "Discrete actions larger than 2 are not supported here."
                        )
                    discrete_actions = np.greater(action[:, counter], 0.0).astype(np.float32)
                original_action.append(discrete_actions)
                counter += 1
            else:
                raise NotImplementedError

        return original_action


def can_import(module_name):
    return not cant_import(module_name)


def cant_import(module_name):
    try:
        importlib.import_module(module_name)
        return False
    except ImportError:
        return True

