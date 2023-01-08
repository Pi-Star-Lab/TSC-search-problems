import os
import time
from os.path import join
from models.memory import Memory
from concurrent.futures.process import ProcessPoolExecutor
import pickle

class UncertainityCurriculum:
    def __init__(self, model_name, sample_problems, ncpus=1, \
            state_generator = None, initial_budget=2000, gradient_steps=10):
        self._model_name = model_name
        self._sample_problems = sample_problems

        self._ncpus = ncpus
        self._initial_budget = initial_budget
        self._max_budget = initial_budget * 2
        self._gradient_steps = gradient_steps
        self._batch_size = 32

        self._state_gen = state_generator
        self._kmax = 10

        # store number of expansions and performance
        self._test_set = pickle.load(open('stp_3_times_3_test', 'rb')) ##TAKE THIS OUTSIDE
        self._expansions = [0]
        self._performance = [0] ## accuracy

        ### global variables could be taken as input
        self._states_per_difficulty = 256
        self._network_confidence = {} #TODO: figure out this part
        self._percentage_store = 0.05 # 5 percent

        self._log_folder = 'training_logs/'
        self._models_folder = 'trained_models_online/' + self._model_name + "_curriculum"

        if not os.path.exists(self._models_folder):
            os.makedirs(self._models_folder)

        if not os.path.exists(self._log_folder):
            os.makedirs(self._log_folder)

    def solve(self, states, planner, nn_model, budget, update:bool):
        """
        states: an iterable object containing name, state
        returns:
        solved, expanded, generate
        """
        batch_problems = {}
        memory = Memory()
        current_solved_puzzles = set()
        number_solved = 0
        total_expanded = 0
        total_generated = 0

        for name, state in states.items():

            # whats name of a state?
            batch_problems[name] = state

            if len(batch_problems) < self._batch_size and \
                    len(states) - number_solved  > self._batch_size:
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

                #perhaps do not count current solved puzzles
                if has_found_solution and puzzle_name not in current_solved_puzzles:
                    number_solved += 1

                    #do we really need this?
                    current_solved_puzzles.add(puzzle_name)

            if memory.number_trajectories() > 0 and update:
                for _ in range(self._gradient_steps):
                    loss = nn_model.train_with_memory(memory)
                    if _ % 10 == 0:
                        print('Iteration: {} Loss: {}'.format(_, loss))
                memory.clear()
                nn_model.save_weights(join(self._models_folder, 'model_weights'))

            batch_problems.clear()
        return (number_solved, total_expanded, total_generated)



    def solve_uniform_online(self, planner, nn_model):
        iteration = 1
        number_solved = 0
        total_expanded = 0
        total_generated = 0
        difficulty = 1
        diameter = 28 ##TODO fix this constant
        budget = self._max_budget

        ## TODO: remove this TMP!

        """
        states = {}
        import sys
        import pickle
        for i in range(self._states_per_difficulty):
            states[i] = self._state_gen(difficulty)
        with open("stp_3_times_3_test", 'wb') as fname:
            pickle.dump(states, fname)
        sys.exit(0)
        """
        while difficulty < diameter:
            number_solved = 0

            states = {}
            for i in range(self._states_per_difficulty):
                states[i] = self._state_gen(difficulty)

            self._network_confidence
            start = time.time()
            number_solved, total_expanded, total_generated = self.solve(states,
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

            print('Percent solved: {}\t Difficulty: {}'.format(number_solved / len(states), difficulty))

            self._expansions.append(total_expanded)
            test_solved, test_expanded, test_generated = self.solve(self._test_set,\
                    planner = planner, nn_model = nn_model, budget = budget, update = False)

            self._performance.append(test_expanded)
            if self.solvable(nn_model, number_solved, total_expanded, total_generated):
                difficulty += 1
            iteration += 1
        self.show_results()

    def solvable(self, nn, number_solved, total_expanded, total_generated): #maybe just use nn

        if number_solved / self._states_per_difficulty > 0.9:
            return True
        else:
            return False
        """
        TODO: write code on content below return statement
        """
        output = nn.multiple_predict(x)
        return True

    def show_results(self):
        print(self._expansions)
        print(self._performance)
