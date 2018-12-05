import logging
import torch

from DQN import DQNHyperparameters, DQNTrainingState
from basic_vizdoom_env import BasicDoomEnv
from my_DQN import MaitlandDQN

if __name__ == "__main__":
    FORMAT = '%(asctime)-15s | %(filename)s:%(lineno)s | %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyper = DQNHyperparameters(batch_size=128)
    env = BasicDoomEnv()
    state = DQNTrainingState(MaitlandDQN, env, device, hyper)

    state.load_model("saved_nets/basic_doom.mod")

    for i in range(10):
        state.run_episode(True)

