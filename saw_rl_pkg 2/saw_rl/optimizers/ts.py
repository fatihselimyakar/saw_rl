"""
SAW Runway Sequencing — Rolling Window Tabu Search
====================================================
Kullanım:
    python ts.py --arr arrivals.csv --dep departures.csv
"""

import argparse
from collections import deque

from saw_rl.optimizers.base_optimizer import BaseOptimizer


class TabuSearch(BaseOptimizer):
    algorithm_name = "Tabu Search"

    def __init__(self, arr_csv, dep_csv, n_window=10, mps_k=6,
                 tabu_tenure=5, max_iters=50):
        self.tabu_tenure = tabu_tenure
        self.max_iters   = max_iters
        super().__init__(arr_csv, dep_csv, n_window, mps_k)

    def optimize_window(self, window, last_ts, last_cat, sched_pos):
        n = len(window)
        if n <= 1:
            return [0]

        current      = list(range(n))
        current_cost = self._eval_sequence(current, window, last_ts, last_cat, sched_pos)
        best         = current.copy()
        best_cost    = current_cost
        tabu_list    = deque(maxlen=self.tabu_tenure)

        for _ in range(self.max_iters):
            best_neighbor      = None
            best_neighbor_cost = float('inf')
            best_move          = None

            for i in range(n):
                for j in range(i + 1, n):
                    neighbor             = current.copy()
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    cost                 = self._eval_sequence(neighbor, window, last_ts, last_cat, sched_pos)
                    move                 = (i, j)
                    is_tabu              = move in tabu_list or (j, i) in tabu_list

                    # Aspiration: global rekoru kıran tabu hamleyi serbest bırak
                    if is_tabu and cost < best_cost:
                        is_tabu = False

                    if not is_tabu and cost < best_neighbor_cost:
                        best_neighbor_cost = cost
                        best_neighbor      = neighbor
                        best_move          = move

            if best_neighbor is None:
                break

            current      = best_neighbor
            current_cost = best_neighbor_cost
            tabu_list.append(best_move)

            if current_cost < best_cost:
                best_cost = current_cost
                best      = current.copy()

        return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling Window Tabu Search")
    parser.add_argument("--arr",         required=True)
    parser.add_argument("--dep",         required=True)
    parser.add_argument("--window",      type=int, default=10)
    parser.add_argument("--mps_k",       type=int, default=6)
    parser.add_argument("--tabu_tenure", type=int, default=5)
    parser.add_argument("--max_iters",   type=int, default=50)
    args = parser.parse_args()

    ts = TabuSearch(
        arr_csv=args.arr, dep_csv=args.dep,
        n_window=args.window, mps_k=args.mps_k,
        tabu_tenure=args.tabu_tenure, max_iters=args.max_iters,
    )
    ts.run_simulation()
