import logging
import random
import torch
from collections import namedtuple

import numpy as np
from gym import Env, Space, spaces
from DQN import DQNTrainingState, DQNHyperparameters
from my_DQN import MaitlandDQN, FCDQN

logger = logging.getLogger(__name__)


class BasicXORActionSpace(spaces.Discrete):
    def __init__(self):
        super().__init__(2)


xor_state = namedtuple("State", ["state", "reward"])


class BasicXORStateSpace(Space):
    def __init__(self):
        super().__init__((2,), np.uint8)
        self.spaces = []
        for a in range(2):
            for b in range(2):
                array = np.zeros([2], dtype=self.dtype)
                array[0] = a
                array[1] = b
                reward = a ^ b
                state = xor_state(array, reward)
                self.spaces.append(state)

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

    def step(self, action):
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
        return None, (self.current_state.reward == action) * 1.0, True, self.current_state

    def reset(self):
        self.current_state = BasicXOREnv.observation_space.sample()
        return self.current_state.state

    def render(self, mode='human'):
        return self.current_state.state



if __name__ == "__main__":
    FORMAT = '%(asctime)-15s | %(filename)s:%(lineno)s | %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyper = DQNHyperparameters(batch_size=128)
    env = BasicXOREnv()
    env.render("human")
    env.reset()
    env.max_steps = 200
    logger.info("Action shape: %s" % (env.action_space.shape,))
    logger.info("Observation Shape: %s" % (env.observation_space.shape,))
    logger.info("Observation space: %s" % env.observation_space.spaces)
    actions = [env.action_space.sample() for _ in range(10)]
    logger.info("Action space: %s" % actions)

    state = DQNTrainingState(FCDQN, env, device, hyper, verbose=True)

    state.train_for_episodes(50000)

    state.save_model("saved_nets/xor_env.mod")


