import os
import time
from os.path import join
from models.memory import Memory
from concurrent.futures.process import ProcessPoolExecutor
import pickle
import random
from curriculum.curriculum import Curriculum

class Bootstrap(Curriculum):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_steps = 10000
        self._min_steps = 5
        self._max_states = self._states_per_itr
        self._number_problems = self._max_states

    def learn_online(self, planner, nn_model):
        iteration = 1
        number_solved = 0
        total_expanded = 0
        total_generated = 0

        budget = self._initial_budget
        memory = Memory()

        current_solved_puzzles = set()

        test_solve = 0
        """
        way to create problems with smaller instance sizes
        TODO: fix this!
        """
        states = {}

        for i in range((self._max_states)):
            steps = self._min_steps + random.random() * (self._max_steps - self._min_steps)
            steps = int(steps)
            states[i] = self._state_gen(steps)

        last_puzzle = list(states)[-1]

        while test_solve < 1: #replacing for comparison
            start = time.time()
            #print("Iteration: {}:".format(iteration)) number_solved = 0

            itr_solved = 0
            batch_problems = {}
            for name, state in states.items():

#                 if name in current_solved_puzzles:
#                     continue

                batch_problems[name] = state

                if len(batch_problems) < self._batch_size and last_puzzle != name:
                    continue

                with ProcessPoolExecutor(max_workers = self._ncpus) as executor:
                    args = ((state, name, budget, nn_model) for name, state in batch_problems.items())
                    results = executor.map(planner.search_for_learning, args)
                for result in results:
                    has_found_solution = result[0]
                    trajectory = result[1]
                    total_expanded += result[2]
                    total_generated += result[3]
                    puzzle_name = result[4]

                    if has_found_solution:
                        memory.add_trajectory(trajectory)

                    if has_found_solution and puzzle_name not in current_solved_puzzles:
                        number_solved += 1
                        itr_solved += 1
                        current_solved_puzzles.add(puzzle_name)

                if memory.number_trajectories() > 0:
                    for _ in range(self._gradient_steps):
                        loss = nn_model.train_with_memory(memory)
                        if _ == 0:
                            pass
                            #print('Loss: ', loss)
                    memory.clear()
                    nn_model.save_weights(join(self._models_folder, 'model_weights'))

                batch_problems.clear()

            end = time.time()
            with open(join(self._log_folder + 'training_bootstrap_' + self._model_name), 'a') as results_file:
                results_file.write(("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format(iteration,
                                                                                 number_solved,
                                                                                 self._number_problems - len(current_solved_puzzles),
                                                                                 budget,
                                                                                 total_expanded,
                                                                                 total_generated,
                                                                                 end-start)))
                results_file.write('\n')

            if itr_solved == 0:
                budget *= 2
                print('Budget: ', budget)

            self._expansions.append(total_expanded)
            test_sol_qual, test_solved, test_expanded, test_generated, _, _ = self.solve(self._test_set,\
                    planner = planner, nn_model = nn_model, budget = self._test_budget, memory = memory, update = False) #TODO: remove this hardcode

            self._test_solution_quality = test_sol_qual
            self._test_expansions = test_expanded

            test_solve = test_solved/len(self._test_set)
            self._time.append(self._time[-1] + (end - start))
            if test_solved == 0:
                self._solution_quality.append(0)
                self._solution_expansions.append(0)
            else:
                self._solution_quality.append(test_sol_qual / test_solved)
                self._solution_expansions.append(test_expanded / test_solved)
            self._performance.append(test_solve)
            print('Training solve: {}%\t Test Solve: {}%'.format(
                number_solved / len(states) * 100, test_solve * 100))

            iteration += 1

        self.print_results()

