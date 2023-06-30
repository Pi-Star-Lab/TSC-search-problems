import os
import time
from os.path import join
from models.memory import Memory
from concurrent.futures.process import ProcessPoolExecutor
import pickle
from curriculum.rw_curriculum import RWCurriculum
from bandit import NonStatBandit
from cmaes_teacher import CMAESTeacher
import numpy as np
from copy import deepcopy

def get_forward_action(state, next_state):
    """
    TODO: write inverse action functions for each domain to avoid this
    """
    actions = state.successors()
    for a in actions:
        state_copy = deepcopy(state)
        state_copy.apply_action(a)
        if state_copy == next_state:
            return a

class LCBCurriculum(RWCurriculum):

    def __init__(self, **kwargs):

        self.goal_state_generator = kwargs['goal_gen']  ##TODO: use same for all Curr approaches
        del kwargs['goal_gen']
        super().__init__(**kwargs)

    def generate_state(self, nn_model):
        # Requries policy to be optimized over Levin loss function
        # IMP
        log_budget = np.log(self._initial_budget)
        log_prob_traj = 0
        depth = 1
        state = self.goal_state_generator()
        while np.log(depth) - log_prob_traj < log_budget:
            prev_state = deepcopy(state)
            state.take_random_action()
            log_act_dist, act_dist, _ = nn_model.predict(np.array([state.get_image_representation()]))
            action = get_forward_action(state, prev_state)
            log_prob_traj += log_act_dist[0][action]
            #print(act_dist[0][action], log_act_dist[0][action], depth, np.exp(log_prob_traj))
            depth += 1
        return prev_state, depth - 1

    def learn_online(self, planner, nn_model):
        iteration = 1
        number_solved = 0
        total_expanded = 0
        total_generated = 0
        budget = self._initial_budget
        test_solve = 0
        memory = Memory()

        while test_solve < 0.9:
            start = time.time()
            number_solved = 0

            states = {}
            difficulties = []
            for i in range(self._states_per_difficulty):
                states[i], difficulty = self.generate_state(nn_model)
                difficulties.append(difficulty)
                #print(states[i], difficulty)
            _, number_solved, total_expanded, total_generated, sol_costs, sol_expansions = self.solve(states,
                        planner=planner, nn_model=nn_model, budget=budget, memory=memory, update=True)

            end = time.time()
            with open(join(self._log_folder + 'training_lcbc_' + self._model_name + "_curriculum"), 'a') as results_file:
                results_file.write(("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format(difficulty,
                                                                                iteration,
                                                                                 number_solved,
                                                                                 3,
                                                                                 budget,
                                                                                 total_expanded,
                                                                                 total_generated,
                                                                                 end-start)))
                results_file.write('\n')



            test_sol_qual, test_solved, test_expanded, test_generated, _, _ = self.solve(self._test_set,
                    planner=planner, nn_model=nn_model, budget=self._test_budget, memory=memory, update=False)

            test_solve = test_solved / len(self._test_set)
            mean_difficulty = sum(difficulties) / len(difficulties)
            print('Iteration: {}\t Train solved: {}\t Test Solved:{}% Mean Difficulty: {}'.format(
                iteration, number_solved / len(states) * 100, test_solve * 100, mean_difficulty))

            self._time.append(self._time[-1] + (end - start))
            self._performance.append(test_solve)
            self._expansions.append(self._expansions[-1] + total_expanded)
            if test_solved == 0:
                self._solution_quality.append(0)
                self._solution_expansions.append(0)
            else:
                self._solution_quality.append(test_sol_qual / test_solved)
                self._solution_expansions.append(test_expanded / test_solved)

            iteration += 1
        self.print_results()
