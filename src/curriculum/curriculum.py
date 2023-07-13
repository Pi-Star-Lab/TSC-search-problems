from abc import ABC, abstractmethod
import os
import pickle
from models.memory import Memory
from concurrent.futures.process import ProcessPoolExecutor

class Curriculum(ABC):
    def __init__(self, num_states, model_name, state_generator, test_set_path,\
            ncpus=1, initial_budget=400, test_budget=400, gradient_steps=10):

        self._model_name = model_name

        self._ncpus = ncpus
        self._test_budget = test_budget
        self._initial_budget = initial_budget
        self._states_per_itr = num_states

        self._gradient_steps = gradient_steps
        self._batch_size = 32

        self._state_gen = state_generator
        self._kmax = 10

        # store number of expansions and performance
        self._test_set = pickle.load(open(test_set_path, 'rb')) ##TAKE THIS OUTSIDE
        self._expansions = [0]
        self._performance = [0] ## accuracy
        self._time = [0]
        self._solution_quality = [0]
        self._solution_expansions = [0]
        self._test_solution_quality = None
        self._test_expansions = None
        self._traj = []


        self._log_folder = 'training_logs/'
        self._models_folder = 'trained_models_online/' + self._model_name + "_curriculum"

        if not os.path.exists(self._models_folder):
            os.makedirs(self._models_folder)

        if not os.path.exists(self._log_folder):
            os.makedirs(self._log_folder)

    def solve(self, states, planner, nn_model, budget, memory, update:bool):
        """
        states: an iterable object containing name, state
        returns:
        number of solved, expanded, generate
        solution costs, expansions per problem
        """
        batch_problems = {}
        current_solved_puzzles = set()
        last_puzzle = list(states)[-1]
        number_solved = 0
        total_expanded = 0
        total_generated = 0
        sum_sol_cost = 0
        sol_costs = []
        sol_expansions = []

        for name, state in states.items():

            # whats name of a state?
            batch_problems[name] = state

            if len(batch_problems) < self._batch_size and \
                    last_puzzle != name:
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
                self._traj.append(trajectory)

                if has_found_solution:
                    #print(trajectory.get_solution_costs())
                    sum_sol_cost += trajectory.get_solution_costs()[-1]
                    memory.add_trajectory(trajectory)
                    sol_costs.append(trajectory.get_solution_costs()[-1])
                    sol_expansions.append(result[2])
                else:
                    sol_costs.append(float("inf"))
                    sol_expansions.append(float("inf"))

                #perhaps do not count current solved puzzles
                if has_found_solution and puzzle_name not in current_solved_puzzles:
                    number_solved += 1

                    #do we really need this?
                    current_solved_puzzles.add(puzzle_name)

            if memory.number_trajectories() > 0 and update:
                for _ in range(self._gradient_steps):
                    loss = nn_model.train_with_memory(memory)
                    if _ % 10 == 0:
                        pass
                        #print('Iteration: {} Loss: {}'.format(_, loss))
                memory.clear()
                nn_model.save_weights(os.path.join(self._models_folder, 'model_weights'))

            batch_problems.clear()
        return (sum_sol_cost, number_solved, total_expanded, total_generated, sol_costs, sol_expansions)

    def get_traj(self):
        traj = self._traj
        self._traj = []
        return traj

    @abstractmethod
    def learn_online(self, planner, nn_model):
        pass

    @staticmethod
    def generate_test_set(state_gen, path, num_samples, max_steps):
        states = {}
        import pickle
        for i in range(num_samples):
            states[i] = state_gen(max_steps)
            print(states[i])
        with open(path, 'wb') as fname:
            pickle.dump(states, fname)

    def print_results(self):
        print("Number Avg Test Expasions:", self._test_expansions / len(self._test_set))
        print("Test Avg Solution Quality:", self._test_solution_quality / len(self._test_set))
        print("Train Expansions:", self._expansions)
        print("Train percent solved:", self._performance)
        print("Train Sol Quality:", self._solution_quality)
        print("Train Expansions:", self._solution_expansions)
        print("Train Cumulative Time:", self._time)
