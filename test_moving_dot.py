import logging

import gym
import gym_moving_dot
import torch

from DQN import DQNTrainingState, DQNHyperparameters
from my_DQN import MaitlandDQN

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    FORMAT = '%(asctime)-15s | %(filename)s:%(lineno)s | %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyper = DQNHyperparameters(batch_size=128)
    env = gym.make("MovingDot-v0")
    env.render("human")
    env.reset()
    env.max_steps = 200
    logger.info("Action shape: %s" % (env.action_space.shape,))
    logger.info("Observation Shape: %s" % (env.observation_space.shape,))

    state = DQNTrainingState(MaitlandDQN, env, device, hyper)
    state.load_model("saved_nets/moving_dot.mod")

    state.run_episode(test=True)

