"""
SAW Runway Sequencing — Unified Benchmark
==========================================
Tum algoritmalari ayni veri seti uzerinde calistirir ve
yan yana karsilastirma tablosu uretir.

Kullanim:
    python benchmark.py --arr data/adsb2_rapor_saw_arrivals.csv --dep data/adsb2_rapor_saw_departures.csv
    python benchmark.py --arr data/adsb2_rapor_saw_arrivals.csv --dep data/adsb2_rapor_saw_departures.csv --skip ga aco
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from saw_rl.optimizers.ts  import TabuSearch
from saw_rl.optimizers.sa  import SimulatedAnnealing
from saw_rl.rl.runway_env  import RunwayEnv


# ──────────────────────────────────────────────────────────────
# FCFS Baseline
# ──────────────────────────────────────────────────────────────
def run_fcfs(arr_csv, dep_csv, n_window, mps_k):
    env = RunwayEnv(arr_csv, dep_csv, n_window=n_window, mps_k=mps_k)
    obs, _ = env.reset()
    done = False
    t0   = time.perf_counter()
    while not done:
        obs, _, terminated, truncated, info = env.step(0)
        done = terminated or truncated
    elapsed = time.perf_counter() - t0
    summary = info.get('episode_summary', {})
    return {
        'algorithm'       : 'FCFS',
        'total_delay_min' : round(summary.get('total_delay_min', 0), 2),
        'avg_delay_per_ac': round(summary.get('avg_delay_min',   0), 3),
        'violations'      : summary.get('violations', 0),
        'n_scheduled'     : summary.get('n_scheduled', env.n_total),
        'elapsed_sec'     : round(elapsed, 2),
    }


# ──────────────────────────────────────────────────────────────
# RL Degerlendirme (GÜNCELLENDİ - VecNormalize ve Best Model Entegrasyonu)
# ──────────────────────────────────────────────────────────────
def run_rl(arr_csv, dep_csv, n_window, mps_k, model_dir='models'):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def mask_fn(env): return env.action_masks()

    # Öncelikli olarak en iyi modeli ara
    best_model_path = os.path.join(model_dir, "best_model", "best_model.zip")
    final_model_path = os.path.join(model_dir, "saw_sequencer_final.zip")
    
    if os.path.exists(best_model_path):
        model_path = best_model_path
    elif os.path.exists(final_model_path):
        model_path = final_model_path
    else:
        raise FileNotFoundError(f"Model bulunamadi. {model_dir} dizinini kontrol et.")

    print(f"      Model yukleniyor: {Path(model_path).name}")

    # 1. Base Env
    base_env = RunwayEnv(arr_csv, dep_csv, n_window=n_window, mps_k=mps_k)
    base_env = ActionMasker(base_env, mask_fn)
    vec_env = DummyVecEnv([lambda: base_env])

    # 2. Normalizasyon verilerini yukle (Körlüğü önler)
    # best_model kullanılıyorsa eşleşen istatistikleri yükle
    best_norm_path  = os.path.join(model_dir, "best_model", "vec_normalize_best.pkl")
    final_norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    if model_path == best_model_path and os.path.exists(best_norm_path):
        norm_path = best_norm_path
    else:
        norm_path = final_norm_path
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False       # Testte yeni veri ogrenme
        vec_env.norm_reward = False    # Odulu normalize etme
    else:
        print(f"      [!] UYARI: Normalizasyon dosyasi ({norm_path}) bulunamadi!")

    # 3. Modeli yukle
    model = MaskablePPO.load(model_path, env=vec_env)

    # 4. Simülasyonu çalıştır (10 kez test edip ortalamasını alırız)
    t0 = time.perf_counter()
    delays, violations = [], []
    
    for _ in range(10):
        obs = vec_env.reset()
        done = False
        while not done:
            masks  = get_action_masks(vec_env)
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            
        # VecEnv info'yu array olarak döner
        real_info = info[0]
        s = real_info.get('episode_summary', {})
        delays.append(s.get('total_delay_min', 0))
        violations.append(s.get('violations', 0))

    elapsed = time.perf_counter() - t0

    return {
        'algorithm'       : 'MaskablePPO (RL)',
        'total_delay_min' : round(float(np.mean(delays)), 2),
        'avg_delay_per_ac': round(float(np.mean(delays)) / max(base_env.unwrapped.n_total, 1), 3),
        'violations'      : int(round(float(np.mean(violations)))),
        'n_scheduled'     : base_env.unwrapped.n_total,
        'elapsed_sec'     : round(elapsed / 10.0, 2), # 10 tur attığımız için ortalama süre
    }


# ──────────────────────────────────────────────────────────────
# Tablo yazdirma
# ──────────────────────────────────────────────────────────────
METRICS = [
    ('total_delay_min',  'Toplam gecikme (dk)',    '{:.1f}',  True),
    ('avg_delay_per_ac', 'Ort. gecikme/ucak (dk)', '{:.3f}',  True),
    ('violations',       'MPS ihlali',             '{:d}',    True),
    ('elapsed_sec',      'Sure (sn)',               '{:.2f}',  False),
]


def print_table(results, fcfs):
    algos  = [fcfs] + results
    names  = [r['algorithm'] for r in algos]
    col_w  = max(26, max(len(n) for n in names) + 2)
    name_w = 28
    div    = '-' * (name_w + col_w * len(algos) + 2)

    print('\n' + '=' * len(div))
    print('  BENCHMARK SONUCLARI -- ' + datetime.now().strftime('%Y-%m-%d %H:%M'))
    print('=' * len(div))
    print('  ' + '{:<{w}}'.format('Metrik', w=name_w) +
          ''.join('{:>{w}}'.format(n, w=col_w) for n in names))
    print('  ' + div)

    for metric_key, label, fmt, lower_is_better in METRICS:
        row      = '  ' + '{:<{w}}'.format(label, w=name_w)
        fcfs_val = fcfs.get(metric_key, 0)
        for r in algos:
            val  = r.get(metric_key, 0)
            cell = fmt.format(val)
            if r['algorithm'] != 'FCFS' and lower_is_better and isinstance(val, (int, float)):
                diff = val - fcfs_val
                sign = 'A' if diff > 0 else 'V'
                pct  = abs(diff / fcfs_val * 100) if fcfs_val else 0
                cell += ' ({}{:.0f}%)'.format(sign, pct)
            row += '{:>{w}}'.format(cell, w=col_w)
        print(row)

    print('  ' + div)
    if results:
        best = min(results, key=lambda r: r['total_delay_min'])
        imp  = (fcfs['total_delay_min'] - best['total_delay_min']) / max(fcfs['total_delay_min'], 1) * 100
        print('\n  En iyi: {} -- FCFS\'e gore {:.1f}% gecikme iyilestirmesi'.format(
              best['algorithm'], imp))
    print()


# ──────────────────────────────────────────────────────────────
# Ana fonksiyon
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='SAW Runway Sequencing -- Unified Benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--arr',    required=True)
    parser.add_argument('--dep',    required=True)
    parser.add_argument('--window', type=int,  default=10)
    parser.add_argument('--mps_k',  type=int,  default=3)
    parser.add_argument('--skip',      nargs='*', default=[])
    parser.add_argument('--out',       default=None)
    parser.add_argument('--quiet',     action='store_true')
    parser.add_argument('--model_dir', default='models')
    args = parser.parse_args()

    skip = {s.lower() for s in (args.skip or [])}
    log  = 0 if args.quiet else 50

    arr_df = pd.read_csv(args.arr)
    dep_df = pd.read_csv(args.dep)
    print('\n' + '=' * 56)
    print('  Veri: ' + Path(args.arr).stem)
    print('  Inis: {} | Kalkis: {} | Toplam: {}'.format(
          len(arr_df), len(dep_df), len(arr_df) + len(dep_df)))
    print('  Window={} | MPS_K={}'.format(args.window, args.mps_k))
    print('=' * 56)

    # 1. FCFS
    fcfs_result = {'algorithm': 'FCFS', 'total_delay_min': 0,
                   'avg_delay_per_ac': 0, 'violations': 0,
                   'n_scheduled': 0, 'elapsed_sec': 0}
    if 'fcfs' not in skip:
        print('\n[1/6] FCFS baseline...')
        fcfs_result = run_fcfs(args.arr, args.dep, args.window, args.mps_k)
        print('      Gecikme: {:.1f} dk | Sure: {:.1f}s'.format(
              fcfs_result['total_delay_min'], fcfs_result['elapsed_sec']))

    # 2-5. Meta-heuristikler
    def _make_ga():
        from saw_rl.optimizers.ga import GeneticAlgorithm
        return GeneticAlgorithm(args.arr, args.dep, args.window, args.mps_k)

    def _make_aco():
        from saw_rl.optimizers.aco import AntColonyOptimization
        return AntColonyOptimization(args.arr, args.dep, args.window, args.mps_k)

    ALGORITHMS = [
        ('ts',  lambda: TabuSearch(args.arr, args.dep, args.window, args.mps_k)),
        ('sa',  lambda: SimulatedAnnealing(args.arr, args.dep, args.window, args.mps_k)),
        ('ga',  _make_ga),
        ('aco', _make_aco),
    ]

    results = []
    for step, (key, factory) in enumerate(ALGORITHMS, start=2):
        if key in skip:
            print('\n[{}/6] {} -- atlandi'.format(step, key.upper()))
            continue
        print('\n[{}/6] {}...'.format(step, key.upper()))
        try:
            result = factory().run_simulation(log_interval=log)
            results.append(result)
            print('      Gecikme: {:.1f} dk | Sure: {:.1f}s'.format(
                  result['total_delay_min'], result['elapsed_sec']))
        except Exception as e:
            print('      HATA: {}'.format(e))

    # 6. RL
    if 'rl' not in skip:
        print('\n[6/6] MaskablePPO (RL) -- model kontrol ediliyor...')
        try:
            rl_result = run_rl(args.arr, args.dep, args.window, args.mps_k, model_dir=args.model_dir)
            results.append(rl_result)
            print('      Gecikme: {:.1f} dk'.format(rl_result['total_delay_min']))
        except Exception as e:
            print('      HATA: {}'.format(e))

    print_table(results, fcfs_result)

    if args.out:
        out_data = {
            'timestamp': datetime.now().isoformat(),
            'config'   : vars(args),
            'fcfs'     : fcfs_result,
            'results'  : results,
        }
        Path(args.out).write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
        print('  Sonuclar kaydedildi: {}\n'.format(args.out))


if __name__ == '__main__':
    main()