"""
SAW Runway Sequencing — Rolling Window Ant Colony Optimization
===============================================================
Kullanım:
    python -m saw_rl.optimizers.aco --arr arrivals.csv --dep departures.csv
"""

import argparse

import numpy as np

from saw_rl.optimizers.base_optimizer import BaseOptimizer
from saw_rl.constants import get_sep


class AntColonyOptimization(BaseOptimizer):
    algorithm_name = "Ant Colony Optimization"

    def __init__(self, arr_csv, dep_csv, n_window=10, mps_k=6,
                 n_ants=20, n_iters=30, alpha=1.0, beta=2.0,
                 evaporation=0.1, Q=100.0):
        self.n_ants      = n_ants
        self.n_iters     = n_iters
        self.alpha       = alpha
        self.beta        = beta
        self.evaporation = evaporation
        self.Q           = Q
        super().__init__(arr_csv, dep_csv, n_window, mps_k)

    def _feasible(self, local_i, window, ant_spos):
        """Bu adımda bu uçak MPS kısıtını karşılıyor mu?"""
        return abs(ant_spos - int(self._fcfs[window[local_i]])) <= self.mps_k

    def optimize_window(self, window, last_ts, last_cat, sched_pos):
        n = len(window)
        if n <= 1:
            return [0]

        pheromone      = np.ones((n, n), dtype=np.float64) * 0.1
        global_best_seq   = None
        global_best_fit   = float('inf')

        for _ in range(self.n_iters):
            # İterasyon içi en iyi — deposit için kullanılır
            iter_best_seq   = None
            iter_best_fit   = float('inf')
            iter_best_delay = float('inf')

            for _ in range(self.n_ants):
                available  = list(range(n))
                sequence   = []
                ant_ts     = last_ts
                ant_cat    = last_cat
                ant_spos   = sched_pos
                ant_delay  = 0.0

                for pos in range(n):
                    # FIX 1: Önce feasible adayları filtrele (RunwayEnv action_mask mantığı)
                    # Hiç feasible yoksa tümünü aday kabul et (kısıt mümkün değilse en azından devam et)
                    feasible_opts = [i for i in available if self._feasible(i, window, ant_spos)]
                    candidates    = feasible_opts if feasible_opts else available

                    probs = []
                    for local_i in candidates:
                        real_idx = window[local_i]
                        earliest = self._ts[real_idx]
                        sep      = get_sep(ant_cat, self._catstr[real_idx])
                        sched    = max(earliest, ant_ts + sep)
                        delay    = max(0.0, (sched - earliest) / 60.0)
                        eta      = 1.0 / (delay + 1e-6)
                        tau      = pheromone[pos][local_i]
                        probs.append((tau ** self.alpha) * (eta ** self.beta))

                    probs        = np.array(probs)
                    probs        = probs / probs.sum() if probs.sum() > 0 \
                                   else np.ones(len(candidates)) / len(candidates)
                    chosen_local = int(np.random.choice(candidates, p=probs))

                    sequence.append(chosen_local)
                    available.remove(chosen_local)

                    real_idx  = window[chosen_local]
                    sep       = get_sep(ant_cat, self._catstr[real_idx])
                    sched_t   = max(self._ts[real_idx], ant_ts + sep)
                    ant_delay += max(0.0, (sched_t - self._ts[real_idx]) / 60.0)
                    ant_ts    = sched_t
                    ant_cat   = self._catstr[real_idx]
                    ant_spos += 1

                fit = self._eval_sequence(sequence, window, last_ts, last_cat, sched_pos)

                # FIX 2: İterasyon içi best'i global best'ten bağımsız tut
                if fit < iter_best_fit:
                    iter_best_fit   = fit
                    iter_best_seq   = sequence.copy()
                    iter_best_delay = ant_delay

                if fit < global_best_fit:
                    global_best_fit = fit
                    global_best_seq = sequence.copy()

            # FIX 3: Deposit — iterasyon içi cezasız gecikme kullan
            # (global best yerine — çoğu iter'de global best kırılmaz → deposit sıfır oluyordu)
            pheromone *= (1.0 - self.evaporation)
            if iter_best_seq is not None:
                deposit = self.Q / (1.0 + iter_best_delay)
                for pos, local_i in enumerate(iter_best_seq):
                    pheromone[pos][local_i] += deposit

        return global_best_seq if global_best_seq else list(range(n))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rolling Window Ant Colony Optimization")
    parser.add_argument("--arr",         required=True)
    parser.add_argument("--dep",         required=True)
    parser.add_argument("--window",      type=int,   default=10)
    parser.add_argument("--mps_k",       type=int,   default=6)
    parser.add_argument("--n_ants",      type=int,   default=20)
    parser.add_argument("--n_iters",     type=int,   default=30)
    parser.add_argument("--alpha",       type=float, default=1.0)
    parser.add_argument("--beta",        type=float, default=2.0)
    parser.add_argument("--evaporation", type=float, default=0.1)
    parser.add_argument("--Q",           type=float, default=100.0)
    args = parser.parse_args()

    aco = AntColonyOptimization(
        arr_csv=args.arr, dep_csv=args.dep,
        n_window=args.window, mps_k=args.mps_k,
        n_ants=args.n_ants, n_iters=args.n_iters,
        alpha=args.alpha, beta=args.beta,
        evaporation=args.evaporation, Q=args.Q,
    )
    aco.run_simulation()
