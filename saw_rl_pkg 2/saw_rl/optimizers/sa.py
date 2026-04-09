"""
SAW Runway Sequencing — Rolling Window Simulated Annealing
===========================================================
Kullanım:
    python sa.py --arr arrivals.csv --dep departures.csv
"""

import argparse
import math
import random

from saw_rl.optimizers.base_optimizer import BaseOptimizer


class SimulatedAnnealing(BaseOptimizer):
    algorithm_name = "Simulated Annealing"

    def __init__(self, arr_csv, dep_csv, n_window=10, mps_k=6,
                 t_start=1000.0, t_min=0.1, alpha=0.90, max_iter=50):
        self.t_start  = t_start
        self.t_min    = t_min
        self.alpha    = alpha
        self.max_iter = max_iter
        super().__init__(arr_csv, dep_csv, n_window, mps_k)

    def optimize_window(self, window, last_ts, last_cat, sched_pos):
        n = len(window)
        if n <= 1:
            return [0]

        current      = list(range(n))
        current_e    = self._eval_sequence(current, window, last_ts, last_cat, sched_pos)
        best         = current.copy()
        best_e       = current_e
        T            = self.t_start

        while T > self.t_min:
            for _ in range(self.max_iter):
                neighbor       = current.copy()
                i, j           = random.sample(range(n), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbor_e     = self._eval_sequence(neighbor, window, last_ts, last_cat, sched_pos)
                delta          = neighbor_e - current_e

                # Daha iyiyse her zaman, daha kötüyse T'ye bağlı olasılıkla kabul et
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current   = neighbor
                    current_e = neighbor_e
                    if current_e < best_e:
                        best   = current.copy()
                        best_e = current_e

            T *= self.alpha

        return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling Window Simulated Annealing")
    parser.add_argument("--arr",      required=True)
    parser.add_argument("--dep",      required=True)
    parser.add_argument("--window",   type=int,   default=10)
    parser.add_argument("--mps_k",    type=int,   default=6)
    parser.add_argument("--t_start",  type=float, default=1000.0)
    parser.add_argument("--t_min",    type=float, default=0.1)
    parser.add_argument("--alpha",    type=float, default=0.90)
    parser.add_argument("--max_iter", type=int,   default=50)
    args = parser.parse_args()

    sa = SimulatedAnnealing(
        arr_csv=args.arr, dep_csv=args.dep,
        n_window=args.window, mps_k=args.mps_k,
        t_start=args.t_start, t_min=args.t_min,
        alpha=args.alpha, max_iter=args.max_iter,
    )
    sa.run_simulation()
