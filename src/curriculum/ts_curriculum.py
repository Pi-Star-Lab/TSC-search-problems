import os
import time
from os.path import join
from models.memory import Memory
from concurrent.futures.process import ProcessPoolExecutor
import pickle
from curriculum.rw_curriculum import RWCurriculum
from bandit import NonStatBandit

class TSCurriculum(RWCurriculum):

    def learn_online(self, planner, nn_model):
        iteration = 1
        number_solved = 0
        total_expanded = 0
        total_generated = 0
        budget = self._initial_budget
        test_solve = 0
        memory = Memory()
        teacher = NonStatBandit(200)
        ## TODO: remove this TMP!

        while test_solve < 0.9:
            start = time.time()
            number_solved = 0

            states = {}
            difficulties = teacher.get_actions(self._states_per_difficulty)
            for i, difficulty in enumerate(difficulties):
                states[i] = self._state_gen(difficulty)

            _, number_solved, total_expanded, total_generated = self.solve(states,
                        planner=planner, nn_model=nn_model, budget=budget, memory = memory, update=True)

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



            test_sol_qual, test_solved, test_expanded, test_generated = self.solve(self._test_set,\
                    planner = planner, nn_model = nn_model, budget = self._test_budget, memory = memory, update = False)

            test_solve = test_solved / len(self._test_set)
            print('Train solved: {}\t Test Solved:{}% Difficulty: {}'.format(
                number_solved / len(states) * 100, test_solve * 100, difficulty))

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
            teacher.batch_update(difficulties, rewards)
            iteration += 1
        self.print_results()

