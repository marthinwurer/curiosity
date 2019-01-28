import random

import numpy as np
from gym import Env, Space, spaces
from vizdoom import *

from myenv import MyEnv
from utilites import to_torch_channels


class BasicDoomEnv(MyEnv):
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]
    action_names = ["Shoot", "Left", "Right"]

    def __init__(self, frame_skips=1):
        self.game = DoomGame()
        self.game.load_config("../ViZDoom/scenarios/basic.cfg")
        self.game.init()
        self.frame_skips = frame_skips

        self.state = self.game.get_state()

    def get_screen(self):
        return self.game.get_state().screen_buffer

    def step(self, action):
        action = BasicDoomEnv.actions[action]
        reward = self.game.make_action(action, self.frame_skips)
        done = self.game.is_episode_finished()
        if not done:
            self.state = self.game.get_state()
        # else:
        #     reward = self.game.get_total_reward()
        img = self.state.screen_buffer
        misc = self.state.game_variables

        # scale the reward
        reward = reward / 100

        return (img, reward, done, misc)

    def reset(self):
        self.game.new_episode()
        self.state = self.game.get_state()
        img = self.state.screen_buffer
        return img

    def get_observation_shape(self):
        return self.get_screen().shape

    def get_num_actions(self):
        return len(self.actions)

    def get_action_name(self, action):
        return self.action_names[action]

    def get_frame_rate(self):
        return 35

    def set_frame_skips(self, frame_skips):
        self.frame_skips = frame_skips


class BasicDoomActionSpace(spaces.Discrete):
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]
    action_names = ["Shoot", "Left", "Right"]
    action_map = None

    def __init__(self):
        super().__init__(len(BasicDoomActionSpace.actions))

    def contains(self, x):
        # mostly taken from spaces.discrete
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        elif isinstance(x, str):
            return x in self.action_map
        else:
            return False

        return 0 <= as_int < self.n

    @classmethod
    def init_action_map(cls):
        # init the action map
        if cls.action_map is None:
            cls.action_map = {}
            for index, name in enumerate(cls.action_names):
                cls.action_map[index] = cls.actions[index]
                cls.action_map[name] = cls.actions[index]


# Init the basic doom action space
BasicDoomActionSpace.init_action_map()


class BasicDoomObservationSpace(Space):

    def __init__(self, shape):
        super().__init__(shape=shape, dtype=np.uint8)

    def sample(self):
        return np.array(np.random.random_sample(self.shape) * 255, dtype=self.dtype)

    def contains(self, x):
        return x.shape == self.shape and x.dtype == self.dtype


class GymBasicDoomEnv(Env):

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def __init__(self, frame_skips):
        self.game = DoomGame()
        self.game.load_config("../ViZDoom/scenarios/basic.cfg")
        self.game.init()
        self.frame_skips = frame_skips

        self.state = self.game.get_state()

        screen_shape = (
            self.game.get_screen_height(),
            self.game.get_screen_width(),
            self.game.get_screen_channels(),
        )

        self.action_space = BasicDoomActionSpace()
        self.observation_space = BasicDoomObservationSpace(screen_shape)

    def get_screen(self):
        img = self.state.screen_buffer
        img = to_torch_channels(img)
        return img

    def step(self, action: int):
        action = self.action_space.action_map[action]
        reward = self.game.make_action(action, self.frame_skips)
        done = self.game.is_episode_finished()
        if not done:
            self.state = self.game.get_state()
        # else:
        #     reward = self.game.get_total_reward()
        img = self.get_screen()
        misc = self.state.game_variables

        # scale the reward
        reward = reward / 100

        return (img, reward, done, misc)

    def reset(self):
        self.game.new_episode()
        self.state = self.game.get_state()
        return self.get_screen()

    def render(self, mode='human'):
        if mode == "rgb_array":
            screen = self.get_screen()
            return screen

