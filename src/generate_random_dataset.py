import argparse
from curriculum.curriculum import Curriculum
from main import get_state_generator
from domains.sliding_tile_puzzle import SlidingTilePuzzle
from domains.sokoban import Sokoban
from domains.pancake import Pancake
from domains.toh import TOH

def get_domain(domain):

    if domain == "SlidingTile":
        return SlidingTilePuzzle
    if domain == "Pancake":
        return Pancake
    if domain == "TOH":
        return TOH
    raise NotImplementedError

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
    parser.add_argument('-size', action='store', dest='num_samples', type=int, default=256,
                        help='Number of samples')
    parameters = parser.parse_args()

    domain = get_domain(parameters.problem_domain)
    states = {}
    import pickle
    for i in range(parameters.num_samples):
        states[i] = domain.generate_random_state(parameters.problem_size)
        print(states[i])
    with open(parameters.test_path, 'wb') as fname:
        pickle.dump(states, fname)


if __name__ == "__main__":
    main()
