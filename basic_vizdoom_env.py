import numpy as np
from gym import Env, Space
from vizdoom import *

from myenv import MyEnv


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


class BasicDoomActionSpace(Space):
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]
    action_names = ["Shoot", "Left", "Right"]


    def __init__(self):
        super().__init__(shape=(3,), dtype=None)

    def sample(self):
        pass

    def contains(self, x):
        pass


class BasicDoomObservationSpace(Space):

    def __init__(self, shape):
        super().__init__(shape=shape, dtype=np.dtype.unit8)

    def sample(self):
        pass

    def contains(self, x):
        pass


class GymDoomEnv(Env):

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def __init__(self):
        self.action_space = BasicDoomActionSpace()


    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass