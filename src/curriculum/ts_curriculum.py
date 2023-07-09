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

class TSCurriculum(RWCurriculum):

    def init_v(self, length:int, mean:int, std:float):
        v = np.zeros(length)
        samples = np.random.normal(loc=mean, scale=std, size = 10000)
        samples = np.round(samples).astype(int)
        for sample in samples:
            if sample >= length or sample < 0:
                continue
            v[sample] += 1
        return v

    def learn_online(self, planner, nn_model):
        iteration = 1
        number_solved = 0
        total_expanded = 0
        total_generated = 0
        budget = self._initial_budget
        test_solve = 0
        memory = Memory()

        teacher = CMAESTeacher(batch_size=self._states_per_difficulty, mean=4, std=4)
        ## TODO: remove this TMP!

        while test_solve < 1:
            start = time.time()
            number_solved = 0

            states = {}
            difficulties = teacher.get_actions(self._states_per_difficulty)
            for i, difficulty in enumerate(difficulties):
                states[i] = self._state_gen(difficulty)

            _, number_solved, total_expanded, total_generated, sol_costs, sol_expansions = self.solve(states,
                        planner=planner, nn_model=nn_model, budget=budget, memory=memory, update=True)

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

            #TODO: get rewards
            rewards = self.get_rewards(sol_expansions, difficulties) #TODO: make it expanded
            teacher.batch_update(difficulties, rewards)
            print("Difficuties")
            print(difficulties)
            print(rewards)
            iteration += 1
        self.print_results()

    def get_rewards(self, expanded, difficulties):
        rewards = []
        print("expanded: ", expanded, "difficulties: ", len(difficulties))
        for i in range(len(expanded)):
            if expanded[i] == float('inf'):
                rewards.append(-1)
            else:
                rewards.append(expanded[i])
        print("expanded: ", len(expanded), "difficulties: ", len(difficulties), "rewards: ", len(rewards))
        return rewards

