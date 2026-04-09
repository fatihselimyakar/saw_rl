"""
SAW Runway Sequencing — Rolling Window Genetic Algorithm
=========================================================
Gereksinim: pip install deap
Kullanım:
    python ga.py --arr arrivals.csv --dep departures.csv
"""

import argparse
import random

from deap import algorithms, base, creator, tools

from saw_rl.optimizers.base_optimizer import BaseOptimizer


# DEAP global creator — modül yüklendiğinde bir kez tanımla
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


class GeneticAlgorithm(BaseOptimizer):
    algorithm_name = "Genetic Algorithm"

    def __init__(self, arr_csv, dep_csv, n_window=10, mps_k=6,
                 pop_size=50, n_gen=20, cx_prob=0.7, mut_prob=0.2):
        self.pop_size = pop_size
        self.n_gen    = n_gen
        self.cx_prob  = cx_prob
        self.mut_prob = mut_prob
        self._toolbox = base.Toolbox()
        super().__init__(arr_csv, dep_csv, n_window, mps_k)

    def _make_eval(self, window, last_ts, last_cat, sched_pos):
        """DEAP fitness wrapper — tuple döndürmeli."""
        def _eval(individual):
            return (self._eval_sequence(individual, window, last_ts, last_cat, sched_pos),)
        return _eval

    def optimize_window(self, window, last_ts, last_cat, sched_pos):
        n = len(window)
        if n <= 1:
            return [0]

        tb = self._toolbox
        tb.register("indices",    random.sample, range(n), n)
        tb.register("individual", tools.initIterate, creator.Individual, tb.indices)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("mate",       tools.cxPartialyMatched)
        tb.register("mutate",     tools.mutShuffleIndexes, indpb=0.2)
        tb.register("select",     tools.selTournament, tournsize=3)
        tb.register("evaluate",   self._make_eval(window, last_ts, last_cat, sched_pos))

        pop = tb.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        algorithms.eaSimple(
            pop, tb,
            cxpb=self.cx_prob, mutpb=self.mut_prob,
            ngen=self.n_gen, halloffame=hof, verbose=False,
        )
        return list(hof[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling Window Genetic Algorithm")
    parser.add_argument("--arr",      required=True)
    parser.add_argument("--dep",      required=True)
    parser.add_argument("--window",   type=int,   default=10)
    parser.add_argument("--mps_k",    type=int,   default=6)
    parser.add_argument("--pop_size", type=int,   default=50)
    parser.add_argument("--n_gen",    type=int,   default=20)
    parser.add_argument("--cx_prob",  type=float, default=0.7)
    parser.add_argument("--mut_prob", type=float, default=0.2)
    args = parser.parse_args()

    ga = GeneticAlgorithm(
        arr_csv=args.arr, dep_csv=args.dep,
        n_window=args.window, mps_k=args.mps_k,
        pop_size=args.pop_size, n_gen=args.n_gen,
        cx_prob=args.cx_prob, mut_prob=args.mut_prob,
    )
    ga.run_simulation()
