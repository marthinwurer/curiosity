import logging
import random
from collections import namedtuple

import numpy as np
from gym import Env, Space, spaces

logger = logging.getLogger(__name__)


class BasicXORActionSpace(spaces.Discrete):
    def __init__(self):
        super().__init__(2)


xor_state = namedtuple("State", ["state", "reward"])


class BasicXORStateSpace(Space):
    def __init__(self):
        super().__init__((2,), np.uint8)
        self.spaces = [
            xor_state(np.array([0, 0]), [1, 0]),
            xor_state(np.array([0, 1]), [0, 1]),
            xor_state(np.array([1, 0]), [0, 1]),
            xor_state(np.array([1, 1]), [1, 0]),
        ]

    def sample(self):
        return random.sample(self.spaces, 1)[0]

    def contains(self, x):
        return True  # Who really cares?


class BasicXOREnv(Env):
    """
    This environment just returns a random 2-bit state and wants the xor value.
    """
    # Set these in ALL subclasses
    action_space = BasicXORActionSpace()
    observation_space = BasicXORStateSpace()

    def __init__(self):
        self.current_state = BasicXOREnv.observation_space.sample()

    def step(self, action: int):
        """
        Do a single timestep of the environment with the given action
        Args:
            action: Action provided by environment. In this case, an int of the index in the actions list

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # get the reward
        rewards = self.current_state.reward
        reward = rewards[action]
        return None, reward, True, self.current_state

    def reset(self):
        self.current_state = BasicXOREnv.observation_space.sample()
        return self.current_state.state

    def render(self, mode='human'):
        return self.current_state.state


