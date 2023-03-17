import numpy as np
from domains.environment import Environment
import copy

def flip_tail(lst, index):
    return lst[:index] + lst[index:][::-1]

class Pancake(Environment):

    def __init__(self, stack):
        if isinstance(stack, str):
            stack = stack.split(' ')
        self.stack = stack

    # Return "[0,1,2]" from stack '0', '1', '2'
    def __str__(self):
        ans="["
        for n in self.stack:
            ans += str(n) + ","
        ans = ans[:-1] + "]"
        return ans

    def successors(self):
        successors = []
        return list(range(0, len(self.stack) - 1))

    def __eq__(self, other):
        return str(self.stack) == str(other.stack)

    def __hash__(self):
        return hash(str(self.stack))

    def get_image_representation(self): # 1D Tensor! Rather Tensor TODO
        """
        Return the one-hot encoding of the pancake problem
        """
        return np.eye(len(self.stack))[self.stack].reshape(-1)

   # Gap heuristic
    def heuristic_value(self, goal):
        # Currently Goal is assumed to be [0,1,2,3,4,...]
        cost = 0
        for i in range(len(self.stack) - 1):
            if abs(self.stack[i + 1] - self.stack[i]) > 1:
                cost += 1
        if self.stack[0] != 0:
            cost += 1
        return cost

    def as_list(self):
        return self.stack

    def is_solution(self):
        return self == Pancake.get_goal_dummy(len(self.stack))

    """
    @staticmethod
    def parse_state(string):
        stack = Utils.parse_list(string)
        return Pancake(stack,0)
    """

    @staticmethod
    def get_goal_dummy(size):
        stack = [i for i in range(size)]
        return Pancake(stack)

    @staticmethod
    def get_name():
        return "pancake"

    def successors_parent_pruning(self, op):
        return self.successors()

    def apply_action(self, action):
        self.stack = flip_tail(self.stack,action)

    def reset(self):
        pass

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def generate_state(size, steps):
        stack = list(range(size))

        goal_state = Pancake(stack)
        assert(goal_state.is_solution())

        state = goal_state
        for i in range(steps):
            actions = state.successors()
            state.apply_action(np.random.choice(actions))

        return state
