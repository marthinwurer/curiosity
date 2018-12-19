import logging
import math
import random
import time
from itertools import count
from typing import Type

import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple

from gym import Env, Space, spaces
from torch import nn
from torch import optim
from tqdm import tqdm

from basic_vizdoom_env import MyEnv
from utilites import format_screen, log_tensors, image_batch_to_device_and_format, to_batch_shape, to_torch_channels

logger = logging.getLogger(__name__)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQNHyperparameters(object):
    """
    Store the hyperparameters of a DQN run
    """
    def __init__(self, batch_size=128, gamma=0.999, eps_start=0.9, eps_end=0.05, eps_decay=200, target_update=10,
                 memory_size=10000, data_type=torch.float32):
        """

        Args:
            batch_size: How many transitions to train a batch on
            gamma: The discount on the q value of future states
            eps_start: The initial probability of selecting a random action
            eps_end: The final probability of selecting a random action
            eps_decay: The rate of decay of epsilon
            target_update: update the policy net after this many episodes
        """
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TARGET_UPDATE = target_update
        self.memory_size = memory_size
        self.data_type = data_type

    def calc_eps(self, steps):
        """
        Calculate the probability of choosing a random action (epsilon)
        Args:
            steps: the number of steps that the model has taken.
        """
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * steps / self.EPS_DECAY)
        return eps_threshold


class BatchData(object):
    def __init__(self, non_final_next_states, non_final_mask, state_batch, action_batch, reward_batch):
        self.non_final_next_states = non_final_next_states
        self.non_final_mask = non_final_mask
        self.state_batch = state_batch
        self.action_batch = action_batch
        self.reward_batch = reward_batch

    def send_to(self, device):
        if self.non_final_next_states:
            self.non_final_next_states = self.non_final_next_states.to(device)
        self.non_final_mask = self.non_final_mask.to(device)
        self.state_batch = self.state_batch.to(device)
        self.action_batch = self.action_batch.to(device, torch.long)
        self.reward_batch = self.reward_batch.to(device)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # noinspection PyCallingNonCallable
    def get_batch(self, batch_size):
        # get a batch worth of experiences and transpose them into into lists for a batch
        transitions = self.sample(batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # get the mask of all the non-final next states so we can compute their outputs
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.uint8)
        try:
            non_final_next_states_np = np.concatenate([s for s in batch.next_state if s is not None])
            # TODO add this to the nets themselves
            # non_final_next_states = image_batch_to_device_and_format(non_final_next_states, device)
            non_final_next_states = torch.from_numpy(non_final_next_states_np)
        except:
            non_final_next_states = None

        # concatenate all the training data and convert to torch
        state_batch_np = np.concatenate(batch.state)
        state_batch = torch.from_numpy(state_batch_np)
        # TODO this in models
        # state_batch = image_batch_to_device_and_format(state_batch)
        action_batch = torch.from_numpy(np.concatenate(batch.action))
        reward_batch_np = np.concatenate(batch.reward, axis=None)
        reward_batch = torch.from_numpy(reward_batch_np)

        return BatchData(non_final_next_states, non_final_mask, state_batch, action_batch, reward_batch)

    def __len__(self):
        return len(self.memory)


class DQNNet(nn.Module):

    def __init__(self, input_space: Space, action_space: Space, **kwargs):
        super().__init__()
        self.input_shape = input_space.shape
        self.action_shape = action_space.shape

        if isinstance(input_space, spaces.Discrete):
            self.input_shape = (input_space.n,)
        if isinstance(action_space, spaces.Discrete):
            self.action_shape = action_space.n

    def forward(self, *x):
        raise NotImplementedError


class DQNTrainingState(object):
    def __init__(self, model_class: Type[DQNNet], env: Env, device,
                 hyper: DQNHyperparameters, optimizer_type=optim.RMSprop, frameskip=4, verbose=False):
        self.env = env
        self.device = device
        self.hyper = hyper
        self.memory = ReplayMemory(hyper.memory_size)
        self.training_steps = 0
        self.verbose = verbose

        self.frameskip = frameskip

        self.model_class = model_class
        self.policy_net = model_class(env.observation_space, env.action_space)
        self.target_net = model_class(env.observation_space, env.action_space)
        self.policy_net.to(device)
        self.target_net.to(device)
        self.optimizer = optimizer_type(self.policy_net.parameters())

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # noinspection PyCallingNonCallable
    def optimize_model(self) -> float:
        # if we haven't sampled enough to make a full batch, skip optimization for now
        if len(self.memory) < self.hyper.BATCH_SIZE:
            return 0.0

        batch_data = self.memory.get_batch(self.hyper.BATCH_SIZE)
        batch_data.send_to(self.device)

        # Compute V(s_{t+1}) for all next states.
        # get the values of the next states
        next_state_values = torch.zeros(self.hyper.BATCH_SIZE, device=self.device)
        if batch_data.non_final_next_states is not None:
            with torch.no_grad():
                next_state_values[batch_data.non_final_mask] = \
                    self.target_net(batch_data.non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values using the next state's values
        adjusted_next_values = next_state_values * self.hyper.GAMMA
        expected_state_action_values = adjusted_next_values + batch_data.reward_batch

        # zero the gradient before doing the current state
        self.optimizer.zero_grad()

        # compute the actual Q values using the current state
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        # use the gather operation to get a tensor of the actions taken. backprop will go through the gather.
        state_values = self.policy_net(batch_data.state_batch)
        state_action_values = state_values.gather(1, batch_data.action_batch)

        esav = expected_state_action_values.view(self.hyper.BATCH_SIZE, 1)

        # Compute Huber loss
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = state_action_values - esav
        # loss = loss.data.unsqueeze(1)

        # Optimize the model
        # loss.backward()
        state_action_values.backward(loss)
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.abs().mean().detach()

    # noinspection PyCallingNonCallable
    def _take_action(self, action):
        screen, reward, done, misc = self.env.step(action.item())
        reward = torch.tensor([[reward]], dtype=self.hyper.data_type)
        return (screen, reward, done, misc, action)

    def _take_random_action(self):
        action = self.env.action_space.sample()
        action = np.array([[action]])
        return self._take_action(action)

    def _take_net_action(self, state):
        with torch.no_grad():
            formatted_screen = image_batch_to_device_and_format(state, self.device)
            actions = self.policy_net(formatted_screen)
            action = actions.max(1)[1].view(1, 1).cpu()
        return self._take_action(action)

    # noinspection PyCallingNonCallable
    def run_episode(self, test=False) -> (float, int):
        # Initialize the environment and state
        screen = self.env.reset()
        screen = to_torch_channels(screen)
        screen = to_batch_shape(screen)

        total_loss = 0
        total_reward = 0

        # do each step
        for t in count(1):
            last_screen = screen

            # Select and perform an action
            sample = random.random()
            eps = self.hyper.calc_eps(self.training_steps)
            if sample > eps or test:
                screen, reward, done, misc, action = self._take_random_action()
            else:
                screen, reward, done, misc, action = self._take_random_action()
            self.env.render("human")
            total_reward += reward

            reward = torch.tensor([[reward]], dtype=self.hyper.data_type)

            # convert the next state if it exists
            if screen is not None:
                screen = to_torch_channels(screen)
                screen = to_batch_shape(screen)

            # if this is not testing the network, store the data and train
            if not test:
                # Store the transition in memory
                self.memory.push((last_screen, action, screen, reward))

                # Perform one step of the optimization (on the target network)
                total_loss += self.optimize_model()
                self.training_steps += 1
            else:
                # otherwise sleep so we've got a reasonable framerate
                print(reward)
                time.sleep(0.02)

            if done:
                # calculate average loss and return it
                return (total_loss / t, total_reward)

    def train_for_episodes(self, episodes):
        with tqdm(range(episodes), total=episodes, unit="episode") as t:
            for episode in t:
                episode_loss, total_reward = self.run_episode()
                string = 'loss: %f, %f' % (episode_loss, total_reward)
                t.set_postfix_str(string)
                # Update the target network
                if episode % self.hyper.TARGET_UPDATE == 0:
                    logger.debug("Updating target weights")
                    self.update_target()
                    # log_tensors(logger)

    def save_model(self, path):
        logger.info("Saving Model to %s" % path)
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        logger.info("Loading Model from %s" % path)
        self.policy_net.load_state_dict(torch.load(path))
        self.update_target()

















