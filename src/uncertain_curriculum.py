import os
import time
from os.path import join
from models.memory import Memory
from concurrent.futures.process import ProcessPoolExecutor

class UncertainityCurriculum:
    def __init__(self, model_name, sample_problems, ncpus=1, initial_budget=2000, gradient_steps=10):
        self._states = states
        self._model_name = model_name
        self._sample_problems = sample_problems

        self._ncpus = ncpus
        self._initial_budget = initial_budget
        self._gradient_steps = gradient_steps
        self._batch_size = 32

        self._kmax = 10

        self._log_folder = 'training_logs/'
        self._models_folder = 'trained_models_online/' + self._model_name + "_curriculum"

        if not os.path.exists(self._models_folder):
            os.makedirs(self._models_folder)

        if not os.path.exists(self._log_folder):
            os.makedirs(self._log_folder)

    def solve_uniform_online(self, planner, nn_model):
        iteration = 1
        number_solved = 0
        total_expanded = 0
        total_generated = 0

        budget = self._initial_budget
        memory = Memory()
        start = time.time()

        current_solved_puzzles = set()

        while len(current_solved_puzzles) < self._number_problems:
            number_solved = 0

            batch_problems = {}
            for name, state in self._states.items():

#                 if name in current_solved_puzzles:
#                     continue

                # whats name of a state?
                batch_problems[name] = state

                if len(batch_problems) < self._batch_size and self._number_problems - len(current_solved_puzzles) > self._batch_size:
                    continue

                """
                TODO: delete this comment
                1) needs batch_problems
                rest until batch_problems.clear() looks cool and
                """
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

                    #perhaps do not count current solved puzzles
                    if has_found_solution and puzzle_name not in current_solved_puzzles:
                        number_solved += 1
                        current_solved_puzzles.add(puzzle_name)

                if memory.number_trajectories() > 0:
                    for _ in range(self._gradient_steps):
                        loss = nn_model.train_with_memory(memory)
                        print('Loss: ', loss)
                    memory.clear()
                    nn_model.save_weights(join(self._models_folder, 'model_weights'))

                batch_problems.clear()

            end = time.time()
            with open(join(self._log_folder + 'training_bootstrap_' + self._model_name + "_curriculum"), 'a') as results_file:
                results_file.write(("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format(iteration,
                                                                                 number_solved,
                                                                                 self._number_problems - len(current_solved_puzzles),
                                                                                 budget,
                                                                                 total_expanded,
                                                                                 total_generated,
                                                                                 end-start)))
                results_file.write('\n')

            print('Number solved: ', number_solved)

            iteration += 1

