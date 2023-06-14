import numpy as np
from univar_gaussian_strategy import UnivarEvolutionaryStrategy

class UnivarGaussianTeacher:
    def __init__(self, batch_size, mean, std = 2):
        self.batch_size = batch_size
        self.opt = UnivarEvolutionaryStrategy(mean=mean, sigma = std, popsize = batch_size)

    def get_action(self):
        self.opt.ask()
        return  

    def get_actions(self, num:int):
        assert(num == self.batch_size)
        actions = []
        for i in range(num):
            actions.append(self.get_action())
        return actions

    def batch_update(self, actions:np.ndarray, rewards:np.ndarray):
        sols = []
        for i in range(len(actions)):
            sols.append((actions[i], -rewards[i])) # cma es minimizes
        print(len(sols))
        self.cma.tell(sols)
