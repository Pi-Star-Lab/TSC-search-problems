import numpy as np
from scipy.special import softmax

class NonStatBandit:
    def __init__(self, size:int, lr = 1e-2, V = None):

        self.size = size
        self.lr = lr
        if V is None:
            self.V = np.zeros(size)
        else:
            self.V = V

    def get_action(self):
        #Thompson Sampling
        return np.random.choice(range(self.size), size = 1, p = self.V)

    def get_actions(self, num:int):
        #Thompson Sampling
        p = softmax(self.V / 100)
        print(p)
        return np.random.choice(range(self.size), size = num, p = p)

    def update(self, action:int, reward:float):
        self.V[action] = (1 - self.lr) * self.V[action] + self.lr * reward

    def batch_update(self, actions:np.ndarray, rewards:np.ndarray):
        for i in range(len(actions)):
            self.update(actions[i], rewards[i])
