import logging
import random
import time

import gym
import numpy as np
import torch
import unittest

from DQN import DQNTrainingState, DQNHyperparameters
from my_DQN import MaitlandDQN, FCDQN
from xor_env import BasicXOREnv

logger = logging.getLogger(__name__)
device = torch.device("cpu")


def setUpModule():
    FORMAT = '%(asctime)-15s | %(filename)s:%(lineno)s | %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global batch_size
    batch_size = 8


class TestXOREnv(unittest.TestCase):

    def test_train_xor_env(self):
        hyper = DQNHyperparameters(batch_size=batch_size)
        env = BasicXOREnv()
        env.render("human")
        env.reset()
        logger.info("Action shape: %s" % (env.action_space.shape,))
        logger.info("Observation Shape: %s" % (env.observation_space.shape,))
        logger.info("Observation space: %s" % env.observation_space.spaces)
        actions = [env.action_space.sample() for _ in range(10)]
        logger.info("Action space: %s" % actions)

        state = DQNTrainingState(FCDQN, env, device, hyper, verbose=True)

        state.train_for_episodes(200)

        state.save_model("saved_nets/xor_env.mod")

    def test_xor_env(self):
        seed = int(time.time())
        random.seed(seed)
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        hyper = DQNHyperparameters(batch_size=batch_size)
        env = BasicXOREnv()
        env.render("human")
        env.reset()
        logger.info("Action shape: %s" % (env.action_space.shape,))
        logger.info("Observation Shape: %s" % (env.observation_space.shape,))
        logger.info("Observation space: %s" % env.observation_space.spaces)
        actions = [env.action_space.sample() for _ in range(10)]
        logger.info("Action space: %s" % actions)

        state = DQNTrainingState(FCDQN, env, device, hyper, verbose=True)
        state.load_model("saved_nets/xor_env.mod")

        # run 100 tests and tally the results
        num_tests = 100
        total = 0
        for test in range(num_tests):
            loss, episode_reward = state.run_episode(True)
            total += episode_reward

        average_reward = total / num_tests

        logger.info("Average reward: %s" % average_reward)


if __name__ == '__main__':
    unittest.main()
