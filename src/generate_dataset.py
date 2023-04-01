import argparse
from curriculum import Curriculum
from main import get_state_generator
from os.path import join, isfile
from os import listdir
from domains.sokoban import Sokoban

def load_sokoban_states(problems_folder):
    problem = []
    puzzle_files = []
    states = {}
    if isfile(problems_folder):
        puzzle_files.append(problems_folder)
    else:
        puzzle_files = [join(problems_folder, f) for f in listdir(problems_folder) if isfile(join(problems_folder, f))]

    problem_id = 0

    for filename in puzzle_files:
        with open(filename, 'r') as file:
            all_problems = file.readlines()

        for line_in_problem in all_problems:
            if ';' in line_in_problem:
                if len(problem) > 0:
                    puzzle = Sokoban(problem)
                    states['puzzle_' + str(problem_id)] = puzzle

                problem = []
#                 problem_id = line_in_problem.split(' ')[1].split('\n')[0]
                problem_id += 1

            elif '\n' != line_in_problem:
                problem.append(line_in_problem.split('\n')[0])

        if len(problem) > 0:
            puzzle = Sokoban(problem)
            states['puzzle_' + str(problem_id)] = puzzle
    return states

def main():
    """
    Generate Test set
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', action='store', dest='problem_domain',
                        help='Problem domain (Witness or SlidingTile)')

    parser.add_argument('-problem-size', action='store', dest='problem_size', type=int, default=None,
                        help='Size of problem (specific to each domain')
    parser.add_argument('-test-path', action='store', dest='test_path', type=str, default=None,
                        help='Path to the test set (pickle object)')

    parser.add_argument('-s', action='store', dest='max_steps', type=int, default=None,
                        help='Maximum number of random steps to be taken')
    parser.add_argument('-size', action='store', dest='num_samples', type=int, default=256,
                        help='Number of samples')
    parser.add_argument('-state-path', action='store', dest='problem_path', type=str, default=None,
                        help='Path where base problems are stored (for sokoban)')
    parameters = parser.parse_args()

    if parameters.problem_domain == "Sokoban":
        states = load_sokoban_states(parameters.problem_path)
        state_gen = get_state_generator(parameters.problem_domain,\
                            states)
    else:
        state_gen = get_state_generator(parameters.problem_domain,\
                            parameters.problem_size)
    Curriculum.generate_test_set(state_gen, parameters.test_path,
            parameters.num_samples, parameters.max_steps)

if __name__ == "__main__":
    main()
