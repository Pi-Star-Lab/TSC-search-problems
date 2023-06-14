import argparse
from curriculum.curriculum import Curriculum
from main import get_state_generator
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
    parameters = parser.parse_args()

    state_gen = get_state_generator(parameters.problem_domain,\
                        parameters.problem_size)

    Curriculum.generate_test_set(state_gen, parameters.test_path,
            parameters.num_samples, parameters.max_steps)

if __name__ == "__main__":
    main()
