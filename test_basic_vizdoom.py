import logging
import random
import time
import warnings

import gym
import gym_moving_dot
import numpy as np
import torch
import unittest

from tqdm import tqdm

from DQN import DQNTrainingState, DQNHyperparameters
from basic_vizdoom_env import GymBasicDoomEnv
from my_DQN import MaitlandDQN

logger = logging.getLogger(__name__)
device = torch.device("cpu")

warnings.filterwarnings("ignore")

def setUpModule():
    FORMAT = '%(asctime)-15s | %(filename)s:%(lineno)s | %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global batch_size
    batch_size = 128


SAVE_NAME = "saved_nets/basic_doom.mod"


class TestBasicViZDoom(unittest.TestCase):

    def build_env(self):
        env = GymBasicDoomEnv(10)
        env.render("human")
        env.reset()

        return env

    def setup_state(self):
        seed = int(time.time())
        random.seed(seed)
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        hyper = DQNHyperparameters(batch_size=batch_size)
        env = self.build_env()
        logger.info("Action shape: %s" % (env.action_space.shape,))
        logger.info("Observation Shape: %s" % (env.observation_space.shape,))
        actions = [env.action_space.sample() for _ in range(10)]
        logger.info("Action space: %s" % actions)

        state = DQNTrainingState(MaitlandDQN, env, device, hyper, optimizer_type=torch.optim.RMSprop, verbose=True)
        print(state.policy_net)
        return state

    def run_tests(self, state):
        # run 100 tests and tally the results
        num_tests = 100
        total = 0
        for test in tqdm(range(num_tests)):
            loss, episode_reward = state.run_episode(True)
            total += episode_reward

        average_reward = total / num_tests

        logger.info("Average reward: %s" % average_reward)

    def test_train(self):
        state = self.setup_state()

        state.train_for_episodes(200)

        state.save_model(SAVE_NAME)

    def test_multiple_train(self):
        state = self.setup_state()

        state.load_model(SAVE_NAME)
        for i in range(20):
            logger.info("Iteration %s" % i)
            state.train_for_episodes(200)
            self.run_tests(state)
            state.save_model(SAVE_NAME)

    def test_continue_train(self):
        state = self.setup_state()

        state.load_model(SAVE_NAME)
        state.train_for_episodes(200)
        state.save_model(SAVE_NAME)

    def test_run(self):
        state = self.setup_state()
        state.load_model(SAVE_NAME)

        self.run_tests(state)


if __name__ == '__main__':
    unittest.main()

