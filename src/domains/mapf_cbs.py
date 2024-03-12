import numpy as np
from domains.environment import Environment
import copy

class Agent:
    """
    1. This is a class for the agent in MAPF-CBS
    2. The agent has a position and a goal
    3. path to record the path of the agent
    """
    def __init__(self, start_pos, goal):
        self.start_pos = np.array(start_pos)
        self.goal = np.array(goal)
        self.curr_pos = np.array(start_pos)
        self.path = [self.start_pos.tolist()]
    
    def move(self, action):
        self.curr_pos += np.array(action)
        self.path.append(self.curr_pos.tolist())
    
    def has_reached_goal(self):
        return np.array_equal(self.curr_pos, self.goal)
    
    def reset(self):
        self.curr_pos = self.start_pos
        self.path = [self.start_pos.tolist()]

class MapfCBSEnv(Environment):
    """
    1. This is a grid-based environment for MAPF-CBS
    2. There are two agents and corresponding goals
    3. The width and height of the grid is defined by user
    4. The position of the agents and goals are defined by user
    5. Actions are defined as up, down, left, right, and stay
    6. The terminal condition:
        1) The agents reach their goals
        2) The agents collide
    """
    def __init__(self, width, height, agents):
        self.width = width
        self.height = height
        self.agents = agents
    
    def copy(self):
        return copy.deepcopy(self)

    def successors(self):
        pass
    
    def is_solution(self):
        pass
    
    def apply_action(self, action):
        pass
    
    def get_image_representation(self):
        pass
    
    def heuristic_value(self):
        pass
    
    def reset(self):
        pass
