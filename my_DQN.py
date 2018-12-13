import logging

import torch
import torch.nn.functional as F
from torch import nn

from DQN import DQNNet, DQNHyperparameters, DQNTrainingState
from basic_vizdoom_env import BasicDoomEnv
from utilites import GenericConvolutionalEncoder, flat_shape, flatten, GenericFullyConnected

logger = logging.getLogger(__name__)


class MaitlandDQN(DQNNet):

    def __init__(self, input_shape, num_actions, fc_total=128, activation=F.relu):
        super().__init__(input_shape, num_actions)

        self.conv_layers = GenericConvolutionalEncoder(self.input_shape)
        self.activation = activation

        final_shape = flat_shape(self.conv_layers.output_shape)

        self.fc = nn.Linear(final_shape, fc_total)
        self.to_actions = nn.Linear(fc_total, self.action_shape)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.activation(self.fc(flatten(x)))
        x = self.to_actions(x)
        return x


class FCDQN(DQNNet):

    def __init__(self, input_shape, num_actions, fc_total=128, activation=F.relu):
        super().__init__(input_shape, num_actions)

        self.layers = GenericFullyConnected(self.input_shape, fc_total, 3, activation=activation)
        self.activation = activation

        self.to_actions = nn.Linear(fc_total, self.action_shape)

    def forward(self, x):
        x = self.layers(x)
        x = self.to_actions(x)
        return x


if __name__ == "__main__":
    FORMAT = '%(asctime)-15s | %(filename)s:%(lineno)s | %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyper = DQNHyperparameters(batch_size=128)
    env = BasicDoomEnv()
    state = DQNTrainingState(MaitlandDQN, env, device, hyper)

    print(env.get_observation_shape())
    exit()

    state.train_for_episodes(500)

    state.save_model("saved_nets/basic_doom.mod")






