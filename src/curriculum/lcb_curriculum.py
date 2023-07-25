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
from scipy.special import softmax

def norm(ar):
    ar = np.array(ar)
    return ar / sum(ar)

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

    def generate_state(self, nn_model, traj, budget):
        # Requries policy to be optimized over Levin loss function
        # IMP
        log_budget = np.log(budget)
        log_prob_traj = 0
        depth = 1
        state = traj[0]
        print(state)
        while np.log(depth) - log_prob_traj < log_budget:
            prev_state = deepcopy(state)
            if depth < len(traj):
                state = traj[depth]
                predictions = nn_model.predict(np.array([state.get_image_representation()]))
                if len(predictions) == 3: #heuristic function included?
                    log_act_dist, act_dist, _ = predictions
                else:
                    log_act_dist, act_dist = predictions
                action = get_forward_action(state, prev_state)
                log_prob_traj += log_act_dist[0][action]

            else:
                #state.take_random_action()
                back_actions = state.successors()
                ns = [deepcopy(state) for a in back_actions]
                for i in range(len(ns)):
                    ns[i].apply_action(back_actions[i])
                
                log_act_dist, act_dist = [], []
                for s in ns:
                    predictions = nn_model.predict(np.array([s.get_image_representation()]))
                    if len(predictions) == 3: #heuristic function included?
                        log_ad, ad, _ = predictions
                    else:
                        log_ad, ad = predictions
                    log_act_dist.append(log_ad[0])
                    act_dist.append(ad[0])
                """
                predictions = nn_model.predict(np.array([s.get_image_representation() for s in ns]))
                if len(predictions) == 3: #heuristic function included?
                    log_act_dist, act_dist, _ = predictions
                else:
                    log_act_dist, act_dist = predictions
                """
                actions = [get_forward_action(s, state) for s in ns]
                for i in range(len(ns)):
                    print("next state --- state --- action", ns[i], state, state.actions_id[actions[i]])
                    print(act_dist[i], actions[i])
                p = norm([act_dist[i][a] for i,a in enumerate(actions)])
                print([act_dist[i][a] for i,a in enumerate(actions)])
                for i, s in enumerate(ns):
                    print(p[i], s)
                """
                from domains.toh import TOH
                if TOH([0,2,1,3,3,3,3,3,3]) == state and TOH([1,2,1,3,3,3,3,3,3]) == ns[0]:
                    import sys
                    sys.exit(1)
                """
                idx = np.random.choice(range(len(ns)), p = p)
                log_prob_traj += log_act_dist[idx][actions[idx]]
                state = ns[idx]
            #print(state, "Depth:", depth, "Prob Traj:", np.exp(log_prob_traj), "Prob action:", np.exp(log_act_dist[0][action]), "bound:", depth / np.exp(log_prob_traj))
            print(state, "Depth:", depth, "Prob Traj:", np.exp(log_prob_traj), "bound:", depth / np.exp(log_prob_traj))
            #print(act_dist[0][action], log_act_dist[0][action], depth, np.exp(log_prob_traj))
            depth += 1
        print("-" * 100)
        return prev_state, depth - 1

    def learn_online(self, planner, nn_model):
        iteration = 1
        number_solved = 0
        total_expanded = 0
        total_generated = 0
        budget = self._initial_budget
        test_solve = 0
        memory = Memory()
        trajs = None

        while self._time[-1] < self._time_limit:
            start = time.time()
            number_solved = 0

            states = {}
            difficulties = []
            for i in range(self._states_per_difficulty):
                goal = self.goal_state_generator()
                if trajs is not None and trajs[i] is not None:
                    traj = trajs[i].get_states()
                else:
                    traj = []
                traj = [goal] + traj # TODO: handle traj part and append goal to traj too! useful for cases with multiple goals
                states[i], difficulty = self.generate_state(nn_model, traj, budget)
                difficulties.append(difficulty)
                #print(states[i], difficulty)

            self._traj = []
            _, number_solved, total_expanded, total_generated, sol_costs, sol_expansions = self.solve(states,
                        planner=planner, nn_model=nn_model, budget=budget, memory=memory, update=True)

            trajs = self.get_traj()
            #for state in trajs[0].get_states():
            #    print(state)
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



            mean_difficulty = sum(difficulties) / len(difficulties)
            self._time.append(self._time[-1] + (end - start))
            
            """
            test_sol_qual, test_solved, test_expanded, test_generated, _, _ = self.solve(self._test_set,
                    planner=planner, nn_model=nn_model, budget=self._test_budget, memory=memory, update=False)

            self._test_solution_quality = test_sol_qual
            self._test_expansions = test_expanded

            test_solve = test_solved / len(self._test_set)

            self._performance.append(test_solve)
            self._expansions.append(self._expansions[-1] + total_expanded)
            if test_solved == 0:
                self._solution_quality.append(0)
                self._solution_expansions.append(0)
            else:
                self._solution_quality.append(test_sol_qual / test_solved)
                self._solution_expansions.append(test_expanded / test_solved)
            """

            print('Iteration: {}\t Train solved: {}\t Mean Difficulty: {}'.format(
                iteration, number_solved / len(states) * 100, mean_difficulty))
            iteration += 1
            
        test_sol_qual, test_solved, test_expanded, test_generated, _, _ = self.solve(self._test_set,
                planner=planner, nn_model=nn_model, budget=self._test_budget, memory=memory, update=False)

        self._test_solution_quality = test_sol_qual
        self._test_expansions = test_expanded

        test_solve = test_solved / len(self._test_set)
        
        print("Test Solved:", test_solve)
        self._performance.append(test_solve)
        self._expansions.append(self._expansions[-1] + total_expanded)
        if test_solved == 0:
            self._solution_quality.append(0)
            self._solution_expansions.append(0)
        else:
            self._solution_quality.append(test_sol_qual / test_solved)
            self._solution_expansions.append(test_expanded / test_solved)
        self.print_results()
