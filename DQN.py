import logging
import math
import random
from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple
from torch import nn
from torch import optim
from tqdm import tqdm

from basic_vizdoom_env import MyEnv
from utilites import format_screen, log_tensors, image_batch_to_device_and_format, to_batch_shape

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

    def __len__(self):
        return len(self.memory)


class DQNNet(nn.Module):

    # noinspection PyUnusedLocal
    def __init__(self, input_shape, num_actions, **kwargs):
        super().__init__()

    def forward(self, *x):
        raise NotImplementedError


class DQNTrainingState(object):
    def __init__(self, model_class: DQNNet, env: MyEnv, device,
                 hyper: DQNHyperparameters, optimizer_type=optim.RMSprop, frameskip=4):
        self.env = env
        self.device = device
        self.hyper = hyper
        self.memory = ReplayMemory(hyper.memory_size)
        self.training_steps = 0

        self.input_shape = env.get_observation_shape()
        self.num_actions = env.get_num_actions()
        self.frameskip = frameskip

        self.model_class = model_class
        self.policy_net = model_class(self.input_shape, self.num_actions)
        self.target_net = model_class(self.input_shape, self.num_actions)
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

        # get a batch worth of experiences and transpose them into into lists for a batch
        transitions = self.memory.sample(self.hyper.BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # get the mask of all the non-final next states so we can compute their outputs
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = np.concatenate([s for s in batch.next_state
                                           if s is not None])
        non_final_next_states = image_batch_to_device_and_format(non_final_next_states, self.device)

        # concatenate all the training data and convert to torch
        state_batch = np.concatenate(batch.state)
        state_batch = image_batch_to_device_and_format(state_batch, self.device)
        action_batch = torch.from_numpy(np.concatenate(batch.action))
        reward_batch = torch.from_numpy(np.concatenate(batch.reward))
        action_batch = action_batch.to(self.device, torch.long)
        reward_batch = reward_batch.to(self.device, self.hyper.data_type)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        # use the gather operation to get a tensor of the actions taken. backprop will go through the gather.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # get the values of the next states
        next_state_values = torch.zeros(self.hyper.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.hyper.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss

    # noinspection PyCallingNonCallable
    def run_episode(self) -> (float, int):
        # Initialize the environment and state
        screen = self.env.reset()
        screen = to_batch_shape(screen)
        # screen = format_screen(screen, self.device)
        total_loss = 0
        done = False
        frame_reward = 0

        # do each step
        for t in count():
            # Select and perform an action
            sample = random.random()
            if sample > self.hyper.calc_eps(self.training_steps):
                with torch.no_grad():
                    formatted_screen = image_batch_to_device_and_format(screen, self.device)
                    action = self.policy_net(formatted_screen).max(1)[1].view(1, 1).cpu()
            else:
                action = np.array([[random.randrange(self.num_actions)]])

            last_screen = screen

            reward = 0  # The reward for this step
            for frame in range(self.frameskip):
                screen, frame_reward, done, misc = self.env.step(action.item())
                reward += frame_reward
                if done:
                    break
            reward = np.array([[reward]])

            # convert the next state
            screen = to_batch_shape(screen)

            # Store the transition in memory
            self.memory.push((last_screen, action, screen, reward))

            # Perform one step of the optimization (on the target network)
            total_loss += self.optimize_model()
            self.training_steps += 1

            if done:
                # calculate average loss and return it
                return (total_loss / t, frame_reward)

    def train_for_episodes(self, episodes):
        with tqdm(range(episodes), total=episodes, unit="episode") as t:
            for episode in t:
                episode_loss, reward = self.run_episode()
                string = 'loss: %.3f, %.3f' % (episode_loss, reward)
                t.set_postfix_str(string)
                # Update the target network
                if episode % self.hyper.TARGET_UPDATE == 0:
                    logger.debug("Updating target weights")
                    self.update_target()
                    log_tensors(logger)

    def save_model(self, path):
        logger.info("Saving Model to %s" % path)
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        logger.info("Loading Model from %s" % path)
        self.policy_net.load_state_dict(torch.load(path))
        self.update_target()

















