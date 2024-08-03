## Curriculum Generation for Learning Guiding Functions in State-Space Search Algorithms
Implementation of the algorithms described in ["Curriculum Generation for Learning Guiding Functions in State-Space Search Algorithms"](https://ojs.aaai.org/index.php/SOCS/article/view/31546)
by S. Pendurkar, L. Lelis, N. Sturtevant, and G. Sharon, published at Symposium of Combinatorial Search (SoCS) 2024.

## Setup

Create a python==3.10.8 env with [virtualenv](https://virtualenv.pypa.io/en/latest/) or
[conda](https://docs.conda.io/en/latest/).

Install the required packages as
```
pip install -r requirements.txt
```


## Details
PHS is dubbed BFSLevin (see src/bfs_levin.py). The same class is used to implement LTS, the
tree search algorithm that uses a policy to guide search (see the paper "Single-Agent Policy
Tree Search with Guarantees" by L. Orseau, L. Lelis, T. Lattimore, and T. Weber for details).

PHS can be trained for a small set of The Witness puzzles with the following command:

```
src/main.py --learned-heuristic
			-a LevinStar
			-l LevinLoss
			-m model_test_witness
			-p problems/witness/puzzles_3x3/
			-b 2000
			-d Witness
			--learn <learning method>
```
learning methods can be
curr (for RW+ method in paper)
tscl (for TSC method in paper)
bootstrap (for BL method in paper)
orw (for RW in paper)

Here are the options of search algorithms implemented:

AStar (A*, see file src/search/a_star.py)
GBFS (Greedy-Best First Search, see file src/search/gbfs.py)
PUCT (PUCT, see file src/search/puct.py)
LevinStar (PHS, see file src/search/bfs_levin.py)
Levin (LTS, see file src/search/bfs_levin.py)

For experiments that are limited by a fixed time use the `-learn-time-limit' argument. Use the branch time_bound_exp for these experiments. 

# TODO 

Merge time_bound_exp into master allowing users both option in main branch.

