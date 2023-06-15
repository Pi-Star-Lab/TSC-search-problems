import numpy as np
from domains.environment import Environment
import copy

class TOH(Environment):
    """
    4 peg towers of Hannoi
    goal transfer everything to last peg (peg 3)
    """
    def __init__(self, locations):
        """
        integers seperated by commas. Denote the disk location (smallest to largest) (which peg? from 0-3?)
        """
        if isinstance(locations, str):
            locations = locations.split(' ')
        self.loc = locations

    # Return "[0,1,2]" from locations '0', '1', '2'
    def __str__(self):
        ans="["
        for n in self.loc:
            ans += str(n) + ","
        ans = ans[:-1] + "]"
        return ans

    def get_top_disks(self):
        top_disks = [float("inf") for i in range(4)]
        for i, l in enumerate(self.loc):
            top_disks[l] = min(top_disks[l], i)
        return top_disks
    
    def successors(self):
        successors = []
        top_disks = self.get_top_disks()
        for i in range(4):
            for j in range(i+1, 4):
                if i == j:
                    continue
                if top_disks[i] > top_disks[j]:
                    #move j to i
                    successors.append(j * 100 + i)
                elif top_disks[i] < top_disks[j]:
                    #move i to j
                    successors.append(i * 100 + j)
    
        return successors

    def __eq__(self, other):
        return str(self.loc) == str(other.loc)

    def __hash__(self):
        return hash(str(self.loc))

    def get_image_representation(self):  # 1D Tensor! Rather Tensor TODO
        """
        Return the one-hot encoding of the pancake problem
        """
        return np.eye(len(self.loc))[self.loc].reshape(-1)

    def heuristic_value(self, goal): #TODO
        # Currently Goal is assumed to be [0,1,2,3,4,...]
        cost = 0
        for i in range(len(self.stack) - 1):
            if abs(self.stack[i + 1] - self.stack[i]) > 1:
                cost += 1
        if self.stack[0] != 0:
            cost += 1
        return cost

    def as_list(self):
        return self.loc

    def is_solution(self):
        return self == TOH.get_goal_dummy(len(self.loc))

    """
    @staticmethod
    def parse_state(string):
        stack = Utils.parse_list(string)
        return Pancake(stack,0)
    """

    @staticmethod
    def get_goal_dummy(size):
        locs = [3 for i in range(size)]
        return TOH(locs)

    @staticmethod
    def get_name():
        return "towers of hanoi"

    def successors_parent_pruning(self, op):
        return self.successors()

    def apply_action(self, action):
        source, dest = action // 100, action % 100
        top_disks = self.get_top_disks()
        self.loc[top_disks[source]] = dest

    def reset(self):
        pass

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def generate_state(size, steps):
        goal_state = TOH.get_goal_dummy(size)

        assert(goal_state.is_solution())

        state = goal_state
        for i in range(steps):
            actions = state.successors()
            print(state, actions)
            state.apply_action(np.random.choice(actions))

        return state
