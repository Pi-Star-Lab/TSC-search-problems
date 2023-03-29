import os
import time
from os.path import join
from models.memory import Memory
from concurrent.futures.process import ProcessPoolExecutor
import pickle
from curriculum import Curriculum

class UncertainityCurriculum(Curriculum):
    def __init__(self, **kwargs):

        ### global variables could be taken as input
        self._network_confidence = {} #TODO: figure out this part
        self._percentage_store = 0.05 # 5 percent
        super().__init__(**kwargs)
        self._states_per_difficulty = self._states_per_itr

    def learn_online(self, planner, nn_model):
        iteration = 1
        number_solved = 0
        total_expanded = 0
        total_generated = 0
        difficulty = 4
        budget = self._initial_budget
        test_solve = 0
        ## TODO: remove this TMP!

        while test_solve < 0.9:
            start = time.time()
            number_solved = 0

            states = {}
            for i in range(self._states_per_difficulty):
                states[i] = self._state_gen(difficulty)

            _, number_solved, total_expanded, total_generated = self.solve(states,
                        planner=planner, nn_model=nn_model, budget=budget, update=True)

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



            if iteration % 5 == 0:
                test_sol_qual, test_solved, test_expanded, test_generated = self.solve(self._test_set,\
                        planner = planner, nn_model = nn_model, budget = self._test_budget, update = False)

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

            if self.solvable(nn_model, number_solved, total_expanded, total_generated):
                difficulty += 1
            iteration += 1
        self.print_results()

    def solvable(self, nn, number_solved, total_expanded, total_generated): #maybe just use nn

        if number_solved / self._states_per_difficulty > 0.75:
            return True
        else:
            return False
        """
        TODO: write code on content below return statement
        """
        output = nn.multiple_predict(x)
        return True

