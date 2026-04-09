"""
SAW Runway Sequencing — BaseOptimizer
======================================
Tüm meta-heuristik algoritmaların ortak altyapısı.

Her alt sınıf yalnızca optimize_window() metodunu implemente etmeli:

    class MyAlgorithm(BaseOptimizer):
        def optimize_window(self, window, last_ts, last_cat, sched_pos):
            # window: mevcut N uçağın global index array'i
            # last_ts: son planlanan uçağın scheduled timestamp'i (float, unix)
            # last_cat: son planlanan uçağın kategorisi (str, örn. 'A3')
            # sched_pos: şu ana kadar planlanan uçak sayısı
            # return: local index permütasyonu (örn. [2, 0, 1, ...])
            ...
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd

from saw_rl.constants import (
    CAT_MAP, DEFAULT_CAT, DEFAULT_MPS_K, DEFAULT_N_WINDOW,
    get_sep, normalize_cat,
)


class BaseOptimizer(ABC):
    """
    Kayan pencereli meta-heuristik optimizasyon için soyut taban sınıf.

    Ortak sorumluluklar:
        - Veri yükleme (_load_data)
        - Pencere değerlendirme (_eval_sequence)
        - Simülasyon döngüsü (run_simulation)
        - Sonuç raporlama (_print_results)

    Alt sınıfın sorumluluğu:
        - optimize_window(): mevcut penceredeki en iyi sıralamayı bulmak
    """

    algorithm_name: str = "BaseOptimizer"

    def __init__(
        self,
        arr_csv: str,
        dep_csv: str,
        n_window: int = DEFAULT_N_WINDOW,
        mps_k: int = DEFAULT_MPS_K,
    ) -> None:
        self.n_window = n_window
        self.mps_k    = mps_k
        self._load_data(arr_csv, dep_csv)

    # ─────────────────────────────────────────────────────────────
    # Veri yükleme
    # ─────────────────────────────────────────────────────────────
    def _load_data(self, arr_csv: str, dep_csv: str) -> None:
        arr = pd.read_csv(arr_csv, parse_dates=['target_time'])
        dep = pd.read_csv(dep_csv, parse_dates=['target_time'])

        self.df = pd.concat([arr, dep], ignore_index=True)

        if 'category' in self.df.columns:
            self.df['category'] = self.df['category'].fillna(DEFAULT_CAT).apply(normalize_cat)
            self.df['cat_enc']  = self.df['category'].map(CAT_MAP).fillna(CAT_MAP[DEFAULT_CAT]).astype(int)
        else:
            self.df['category'] = DEFAULT_CAT
            self.df['cat_enc']  = CAT_MAP[DEFAULT_CAT]

        self.df = (
            self.df
            .dropna(subset=['target_time'])
            .sort_values('target_time')
            .reset_index(drop=True)
        )
        self.df['fcfs_rank'] = np.arange(len(self.df))
        self.n_total = len(self.df)

        # Numpy cache — pandas her step'te yavaş
        self._ts     = np.array([t.timestamp() for t in self.df['target_time']], dtype=np.float64)
        self._catstr = self.df['category'].values
        self._fcfs   = self.df['fcfs_rank'].values.astype(np.int32)

    # ─────────────────────────────────────────────────────────────
    # Ortak değerlendirme fonksiyonu
    # ─────────────────────────────────────────────────────────────
    def _eval_sequence(
        self,
        local_order: list[int],
        window: np.ndarray,
        last_ts: float,
        last_cat: str,
        sched_pos: int,
        mps_penalty: float = 10_000.0,
    ) -> float:
        """
        Verilen local index permütasyonunun toplam maliyetini hesaplar.

        Args:
            local_order:  Window içi local index sıralaması (örn. [2, 0, 1])
            window:       Global index array'i (_ts, _catstr'a erişmek için)
            last_ts:      Bir önceki uçağın scheduled timestamp'i (unix float)
            last_cat:     Bir önceki uçağın kategorisi
            sched_pos:    Şimdiye kadar planlanan uçak sayısı (MPS hesabı için)
            mps_penalty:  MPS ihlali başına ceza skoru

        Returns:
            Toplam maliyet = gecikme (dakika) + ihlal cezaları
        """
        ts   = last_ts
        cat  = last_cat
        cost = 0.0

        for step, local_i in enumerate(local_order):
            idx = window[local_i]

            if abs((sched_pos + step) - int(self._fcfs[idx])) > self.mps_k:
                cost += mps_penalty

            earliest = self._ts[idx]
            sep      = get_sep(cat, self._catstr[idx])
            sched    = max(earliest, ts + sep)
            cost    += max(0.0, (sched - earliest) / 60.0)

            ts  = sched
            cat = self._catstr[idx]

        return cost

    # ─────────────────────────────────────────────────────────────
    # Alt sınıfın implemente edeceği metot
    # ─────────────────────────────────────────────────────────────
    @abstractmethod
    def optimize_window(
        self,
        window: np.ndarray,
        last_ts: float,
        last_cat: str,
        sched_pos: int,
    ) -> list[int]:
        """
        Mevcut penceredeki en iyi local index permütasyonunu döndür.

        Args:
            window:    Mevcut N uçağın global index array'i
            last_ts:   Son planlanan uçağın scheduled unix timestamp'i
            last_cat:  Son planlanan uçağın RECAT kategorisi
            sched_pos: Şimdiye kadar planlanan toplam uçak sayısı

        Returns:
            local index sıralaması — ilk eleman bir sonraki planlanan uçak
        """
        ...

    # ─────────────────────────────────────────────────────────────
    # Simülasyon döngüsü
    # ─────────────────────────────────────────────────────────────
    def run_simulation(self, log_interval: int = 50) -> dict:
        """
        Tüm uçakları kayan pencereli optimize_window çağrısıyla planlar.

        Args:
            log_interval: Kaç uçakta bir ara log basılsın (0 = sessiz)

        Returns:
            Sonuç dict'i: total_delay_min, avg_delay_per_ac, violations, elapsed_sec
        """
        print(f"\n[*] {self.algorithm_name} — Simülasyon başlıyor "
              f"({self.n_total} uçak, pencere={self.n_window}, MPS_K={self.mps_k})")

        scheduled   : list[int] = []
        done_mask               = np.zeros(self.n_total, dtype=bool)
        last_ts                 = self._ts[0]
        last_cat                = DEFAULT_CAT
        total_delay             = 0.0
        violations              = 0
        t_start                 = time.perf_counter()

        while len(scheduled) < self.n_total:
            remaining = np.where(~done_mask)[0]
            if len(remaining) == 0:
                break

            order   = np.argsort(self._ts[remaining], kind='stable')
            window  = remaining[order[:self.n_window]]
            sched_pos = len(scheduled)

            best_seq         = self.optimize_window(window, last_ts, last_cat, sched_pos)
            chosen_local_idx = best_seq[0]
            idx              = window[chosen_local_idx]

            earliest = self._ts[idx]
            sep      = get_sep(last_cat, self._catstr[idx])
            sched_ts = max(earliest, last_ts + sep)
            delay    = max(0.0, (sched_ts - earliest) / 60.0)

            if abs(sched_pos - int(self._fcfs[idx])) > self.mps_k:
                violations += 1

            done_mask[idx] = True
            scheduled.append(idx)
            last_ts      = sched_ts
            last_cat     = self._catstr[idx]
            total_delay += delay

            if log_interval and len(scheduled) % log_interval == 0:
                print(f"    [{len(scheduled):>4}/{self.n_total}] gecikme: {total_delay:.1f} dk")

        elapsed = time.perf_counter() - t_start
        results = {
            'algorithm'       : self.algorithm_name,
            'total_delay_min' : round(total_delay, 2),
            'avg_delay_per_ac': round(total_delay / max(self.n_total, 1), 3),
            'violations'      : violations,
            'n_scheduled'     : len(scheduled),
            'elapsed_sec'     : round(elapsed, 2),
        }
        self._print_results(results)
        return results

    # ─────────────────────────────────────────────────────────────
    # Raporlama
    # ─────────────────────────────────────────────────────────────
    def _print_results(self, r: dict) -> None:
        bar = "=" * 56
        print(f"\n{bar}")
        print(f"  SONUÇLAR — {r['algorithm']}")
        print(bar)
        print(f"  Toplam gecikme   : {r['total_delay_min']:.1f} dk")
        print(f"  Ort. gecikme     : {r['avg_delay_per_ac']:.3f} dk/uçak")
        print(f"  MPS ihlali       : {r['violations']}")
        print(f"  Planlanan uçak   : {r['n_scheduled']}/{self.n_total}")
        print(f"  Süre             : {r['elapsed_sec']:.1f} sn")
        print(f"{bar}\n")
