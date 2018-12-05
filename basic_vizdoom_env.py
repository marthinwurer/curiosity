from vizdoom import *


class MyEnv(object):
    """
    A basic environment because OpenAI Gym is too much pain
    """

    def get_observation_shape(self) -> tuple:
        """
        Get the shape of the observation that the environment returns
        """
        raise NotImplementedError

    def get_num_actions(self) -> int:
        """
        Get the number of actions that can be made in this environment
        """
        raise NotImplementedError

    def get_action_name(self, action):
        """
        Return the string value of the name of the action
        Args:
            action:

        Returns:

        """
        raise NotImplementedError
    def reset(self) -> object:
        """
        Reset this environment to the base state, return the starting observation
        Returns:
            observation (object): agent's observation of the current environment
        """
        raise NotImplementedError

    def step(self, action) -> (object, float, bool, dict):
        """
        Do a single timestep of the environment with the given action
        Args:
            action: Action provided by environment. In this case, an int of the index in the actions list

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def get_frame_rate(self):
        """
        Return the number of frames that occur in a second for the game environment
        Returns:

        """
        raise NotImplementedError

    def set_frame_skips(self, frame_skips):
        raise NotImplementedError


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
        else:
            reward = self.game.get_total_reward()
        img = self.state.screen_buffer
        misc = self.state.game_variables

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

