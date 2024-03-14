class Trajectory(object):

    def __init__(self, initial_state):
        self.current_state = initial_state
        self.transitions = []
        self.reward = None
        self.done = False
        self.actions = []

    def update(self, next_state, action, log_reward, done):
        self.transitions.append(
            [self.current_state, next_state, action, log_reward, done]
        )
        self.current_state = next_state
        self.actions.append(action)
        self.done = done
        self.log_reward = log_reward

    def update_reward(self, log_reward):
        self.transitions[-1][-2] = log_reward
        self.log_reward = log_reward


class SimpleTrajectory(object):

    def __init__(self):
        self.actions = []
        self.log_reward = None

    def update(self, action, log_reward):
        self.actions.append(action)
        self.log_reward = log_reward
