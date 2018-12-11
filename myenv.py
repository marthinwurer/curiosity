
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

