import logging

import torch

import gym

from DQN import DQNHyperparameters, DQNTrainingState
from my_DQN import MaitlandDQN

if __name__ == "__main__":
    FORMAT = '%(asctime)-15s | %(filename)s:%(lineno)s | %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyper = DQNHyperparameters(batch_size=128)
    env = gym.make('FrozenLake-v0')
    env.reset()
    env.render()

    state = DQNTrainingState(MaitlandDQN, env, device, hyper)

    state.train_for_episodes(500)
