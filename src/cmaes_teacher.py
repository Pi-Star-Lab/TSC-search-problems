import numpy as np
from cmaes import CMA

class CMAESTeacher:
    def __init__(self, batch_size, mean, std = 2):
        self.batch_size = batch_size
        self.init_std = std
        self.popsize = batch_size
        mean = np.array([mean, .0], ndmin=1) ##TODO: remove the padding in cmaes
        print(mean.shape)
        self.cma = CMA(mean=mean, sigma = std, population_size = batch_size)

    def get_action(self):
        #Thompson Sampling
        return max(1, round(self.cma.ask()[0])) #coz of padding: CMA-ES doesn't work in 1D

    def get_actions(self, num:int):
        #Thompson Sampling
        assert(num == self.batch_size)
        actions = []
        for i in range(num):
            actions.append(self.get_action())
        return actions

    def stop(self):
        return True if self.cma._sigma * self.cma._C[0,0] < 0.01 else False

    def batch_update(self, actions:np.ndarray, rewards:np.ndarray):
        sols = []
        for i in range(len(actions)):
            sols.append(([actions[i], 0.0], -rewards[i])) # cma es minimizes
        print(len(sols))
        self.cma.tell(sols)

        ## restart
        if self.stop():
            print("restart")
            self.cma = CMA(mean=self.cma._mean, sigma=self.init_std, population_size=self.popsize)
