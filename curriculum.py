from abc import ABC, abstractmethod
import os
import pickle

class Curriculum(ABC):
    def __init__(self, model_name, state_generator, ncpus=1, \
            initial_budget=2000, gradient_steps=10):

        self._model_name = model_name

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
        self._time = [0]
        self._solution_quality = [0]
        self._solution_expansions = [0]

        self._log_folder = 'training_logs/'
        self._models_folder = 'trained_models_online/' + self._model_name + "_curriculum"

        if not os.path.exists(self._models_folder):
            os.makedirs(self._models_folder)

        if not os.path.exists(self._log_folder):
            os.makedirs(self._log_folder)

    def print_results(self):
        print(self._expansions)
        print(self._performance)
        print(self._solution_quality)
        print(self._solution_expansions)
        print(self._time)
