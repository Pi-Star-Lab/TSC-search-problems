import os
import time
from os.path import join
from models.memory import Memory
from concurrent.futures.process import ProcessPoolExecutor
import pickle
from copy import deepcopy
from curriculum.curriculum import Curriculum

class ORWCurriculum(Curriculum):
    def __init__(self, **kwargs):

        ### global variables could be taken as input
        self.goal_state_generator = kwargs['goal_gen']  ##TODO: use same for all Curr approaches
        del kwargs['goal_gen']
        super().__init__(**kwargs)
        self._states_per_difficulty = self._states_per_itr

    def get_length(self, budget):
        goal = self.goal_state_generator()
        pq = [goal]
        num_expansion = 0
        visited = set()
        while num_expansion < budget:
            state = pq.pop(0)
            visited.add(state)
            num_expansion += 1
            actions = state.successors()
            for a in actions:
                ns = deepcopy(state)
                ns.apply_action(a)
                if ns not in visited:
                    pq.append(ns)
        length = 0
        for i in range(5000):
            goal = self.goal_state_generator()
            state = goal
            while state in visited:
                state.take_random_action()
                length += 1
        return int(length / 5000)

    def learn_online(self, planner, nn_model):
        iteration = 1
        number_solved = 0
        total_expanded = 0
        total_generated = 0
        difficulty = 0
        budget = self._initial_budget
        test_solve = 0
        memory = Memory()
        prev_test_acc = 0
        ## TODO: remove this TMP!

        while test_solve < 1:
            start = time.time()
            number_solved = 0

            difficulty += self.get_length(budget)
            states = {}
            for i in range(self._states_per_difficulty):
                states[i] = self._state_gen(difficulty)

            _, number_solved, total_expanded, total_generated, _, _ = self.solve(states,
                        planner=planner, nn_model=nn_model, budget=budget, memory = memory, update=True)

            if number_solved == 0:
                budget *= 2
                print("New Training Budget:", budget)
            end = time.time()
            with open(join(self._log_folder + 'training_bootstrap_' + self._model_name + "_curriculum"), 'a') as results_file:
                results_file.write(("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format(difficulty,
                                                                                iteration,
                                                                                 number_solved,
                                                                                 3,
                                                                                 budget,
                                                                                 total_expanded,
                                                                                 total_generated,
                                                                                 end-start)))
                results_file.write('\n')

            self._time.append(self._time[-1] + (end - start))

            self._expansions.append(self._expansions[-1] + total_expanded)

            if prev_test_acc > 0.6 or (iteration - 1) % 5 == 0:
                test_sol_qual, test_solved, test_expanded, test_generated, _, _ = self.solve(self._test_set,\
                        planner = planner, nn_model = nn_model, budget = self._test_budget, memory = memory, update = False)

                self._test_solution_quality = test_sol_qual
                self._test_expansions = test_expanded

                test_solve = test_solved / len(self._test_set)
                prev_test_acc = test_solve
                print('Iteration: {}\t Train solved: {}\t Test Solved:{}% Difficulty: {}'.format(
                    iteration, number_solved / len(states) * 100, test_solve * 100, difficulty))

                self._performance.append(test_solve)
                if test_solved == 0:
                    self._solution_quality.append(0)
                    self._solution_expansions.append(0)
                else:
                    self._solution_quality.append(test_sol_qual / test_solved)
                    self._solution_expansions.append(test_expanded / test_solved)
            else:
                print('Iteration: {}\t Train solved: {} Difficulty: {}'.format(
                    iteration, number_solved / len(states) * 100, difficulty))
            iteration += 1
        self.print_results()
