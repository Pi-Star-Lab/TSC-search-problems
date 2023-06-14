import numpy as np
import sys

class UnivarEvolutionaryStrategy():

    def __init__(self, mean, sigma, popsize, elite = 0.5, min_var = 5):

        self.mean = mean
        self.std = sigma
        self.popsize = popsize
        self.elite = elite
        self.min_var = min_var

    def ask(self):

        return np.random.normal(self.mean, self.std, 1)[0]

    def tell(self, solutions):

        assert(len(solutions) == self.popsize)
        xs, ys = [], []
        for sol in solutions:
            xs.append(sol[0])
            ys.append(sol[1])
        ys, xs = (list(t) for t in zip(*sorted(zip(ys, xs))))

        topk = int(self.popsize * self.elite)
        topx = xs[:topk]
        self.mean = np.mean(topx)
        self.std = max(self.min_var, np.std(topx))

    def coverged(self):
        if self.std < 1e-2:
            return True
        return False

def convex(x):
    return (x - 4) ** 2

time = 0
def non_stat_fn(x):
    global time
    dx = 0.1
    target = 4 + time * dx
    print(target)
    fval = (x - target) ** 2
    time += 1
    return fval

if __name__ == "__main__":

    batch_size = 32
    opt_alg = UnivarEvolutionaryStrategy(-10, 20, 32, elite = 0.1)
    fn = convex
    fn = non_stat_fn
    while not opt_alg.coverged():
        xs = [opt_alg.ask() for x in range(batch_size)]
        ys = [fn(x) for x in xs]
        solutions = []
        for i in range(len(xs)):
            solutions.append((xs[i], ys[i]))
        opt_alg.tell(solutions)
        print(opt_alg.mean, opt_alg.std)
