import numpy as np
from abc import ABC, abstractmethod

class Environment(ABC):
    
    @abstractmethod
    def successors(self):
        pass
    
    @abstractmethod
    def is_solution(self):
        pass
    
    @abstractmethod
    def apply_action(self, action):
        pass
    
    @abstractmethod
    def get_image_representation(self):
        pass
    
    @abstractmethod
    def heuristic_value(self):
        pass
    
    def reset(self):
        pass
    
    def copy(self):
        pass

    def take_random_action(self):
        actions = self.successors()
        action = np.random.choice(actions)
        self.apply_action(action)
        return action
