"""
SAW RL — Ortam Doğrulama Scripti
==================================
Kullanım (saw_rl_pkg kökünden):
    python test_env.py                        # sentetik veri ile test
    python test_env.py --pcap data/kayit.pcapng  # gerçek pcap ile tam pipeline
"""

import argparse
import sys
import traceback
from pathlib import Path

PASS = "  ✓"
FAIL = "  ✗"
SKIP = "  –"


def section(title):
    print(f"\n{'─' * 56}")
    print(f"  {title}")
    print(f"{'─' * 56}")


def ok(msg):   print(f"{PASS} {msg}")
def fail(msg): print(f"{FAIL} {msg}")
def skip(msg): print(f"{SKIP} {msg}")


# ─────────────────────────────────────────────────────────────
# 1. Paket importları
# ─────────────────────────────────────────────────────────────
def test_imports():
    section("1. Paket importları")
    checks = [
        ("saw_rl.constants",                   "get_sep, WAKE_SEP, ROT, CAT_MAP, normalize_cat"),
        ("saw_rl.optimizers.base_optimizer",   "BaseOptimizer"),
        ("saw_rl.optimizers.ts",               "TabuSearch"),
        ("saw_rl.optimizers.sa",               "SimulatedAnnealing"),
        ("saw_rl.optimizers.ga",               "GeneticAlgorithm"),
        ("saw_rl.optimizers.aco",              "AntColonyOptimization"),
        ("saw_rl.rl.runway_env",               "RunwayEnv"),
    ]
    all_ok = True
    for module, names in checks:
        try:
            mod = __import__(module, fromlist=names.split(", "))
            for name in names.split(", "):
                getattr(mod, name.strip())
            ok(f"{module}")
        except Exception as e:
            fail(f"{module} — {e}")
            all_ok = False
    return all_ok


# ─────────────────────────────────────────────────────────────
# 2. constants.py doğruluğu
# ─────────────────────────────────────────────────────────────
def test_constants():
    section("2. constants.py doğruluğu")
    from saw_rl.constants import get_sep, normalize_cat, ROT, WAKE_SEP

    cases = [
        (("A5", "A1"), 160),
        (("A3", "A3"),  80),
        (("A1", "A5"),  80),
        (("A4", "A4"), 100),
    ]
    all_ok = True
    for (lead, follow), expected in cases:
        result = get_sep(lead, follow)
        rot    = ROT.get(lead, 55)
        want   = max(WAKE_SEP.get((lead, follow), 80), rot)
        if result == want:
            ok(f"get_sep({lead},{follow}) = {result}s")
        else:
            fail(f"get_sep({lead},{follow}) = {result}s, beklenen {want}s")
            all_ok = False

    norm_cases = [("A3", "A3"), ("a5", "A5"), ("A3MEDIUM", "A3"), ("", "A3"), ("XYZ", "A3")]
    for raw, expected in norm_cases:
        result = normalize_cat(raw)
        if result == expected:
            ok(f"normalize_cat('{raw}') = '{result}'")
        else:
            fail(f"normalize_cat('{raw}') = '{result}', beklenen '{expected}'")
            all_ok = False
    return all_ok


# ─────────────────────────────────────────────────────────────
# 3. Sentetik veri üret
# ─────────────────────────────────────────────────────────────
def make_synthetic_csvs(out_dir: Path):
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    np.random.seed(42)
    base = datetime(2024, 3, 15, 8, 0, 0)
    CATS = ['A1', 'A2', 'A3', 'A3', 'A3', 'A4', 'A5']

    def make_df(n, phase):
        times = [base + timedelta(seconds=i * 90 + int(np.random.randint(-30, 30))) for i in range(n)]
        cats  = list(np.random.choice(CATS, n))
        return pd.DataFrame({
            'hex'          : [f'{int(np.random.randint(0x400000, 0xFFFFFF)):06x}' for _ in range(n)],
            'flight'       : [f'TK{int(np.random.randint(100, 999))}' for _ in range(n)],
            'target_time'  : times,
            'last_seen'    : [t - timedelta(seconds=int(np.random.randint(60, 300))) for t in times],
            'category'     : cats,
            'dist_km'      : np.round(np.random.uniform(5, 80, n), 2),
            'alt_last'     : np.round(np.random.uniform(500, 8000, n), 0),
            'alt_first'    : np.round(np.random.uniform(8000, 35000, n), 0),
            'gs_avg'       : np.round(np.random.uniform(150, 280, n), 1),
            'baro_rate_med': np.round(np.random.uniform(-2000, -200, n), 0),
            'hdg_diff'     : np.round(np.random.uniform(0, 30, n), 1),
            'phase'        : phase,
            'rssi_last'    : np.round(np.random.uniform(-25, -5, n), 1),
            'nac_p_last'   : np.random.randint(7, 11, n),
            'track_last'   : np.round(np.random.uniform(0, 360, n), 1),
            'lat_last'     : np.round(np.random.uniform(40.5, 41.2, n), 4),
            'lon_last'     : np.round(np.random.uniform(28.8, 29.8, n), 4),
        })

    arr_path = out_dir / "test_arrivals.csv"
    dep_path = out_dir / "test_departures.csv"
    make_df(30, 'ARRIVAL').to_csv(arr_path,  index=False)
    make_df(20, 'DEPARTURE').to_csv(dep_path, index=False)
    return arr_path, dep_path


# ─────────────────────────────────────────────────────────────
# 4. RunwayEnv testi
# ─────────────────────────────────────────────────────────────
def test_runway_env(arr_path, dep_path):
    section("4. RunwayEnv")
    try:
        import numpy as np
        from saw_rl.rl.runway_env import RunwayEnv

        env1 = RunwayEnv(str(arr_path), str(dep_path), n_window=10, mps_k=6)
        env2 = RunwayEnv(str(arr_path), str(dep_path), n_window=5,  mps_k=3)

        # Global mutasyon bug kontrolü
        assert env1._obs_hi[7] != env2._obs_hi[7] or env1.n_total == env2.n_total, \
            "_obs_hi global mutasyon bug'ı hâlâ var!"
        ok("Global _obs_hi mutasyon bug'ı yok")

        obs, info = env1.reset()
        assert obs.shape == (10 * 9,), f"Obs shape yanlış: {obs.shape}"
        ok(f"reset() → obs.shape={obs.shape}")

        mask = env1.action_masks()
        assert mask.shape == (10,)
        assert mask.any(), "Tüm action'lar maskelenmiş!"
        ok(f"action_masks() → {mask.sum()} geçerli action")

        steps = 0
        done  = False
        total_reward = 0.0
        while not done:
            valid = [i for i, m in enumerate(env1.action_masks()) if m]
            obs, rew, terminated, truncated, info = env1.step(valid[0])
            total_reward += rew
            done = terminated or truncated
            steps += 1

        ok(f"FCFS episode tamamlandı: {steps} adım, reward={total_reward:.1f}")
        summary = info.get('episode_summary', {})
        ok(f"Toplam gecikme: {summary.get('total_delay_min', '?'):.1f} dk | "
           f"İhlal: {summary.get('violations', '?')}")
        return True

    except Exception as e:
        fail(f"RunwayEnv hatası: {e}")
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────
# 5. Meta-heuristik smoke test (küçük pencere, az iterasyon)
# ─────────────────────────────────────────────────────────────
def test_optimizers(arr_path, dep_path):
    section("5. Optimizer smoke testleri")
    from saw_rl.optimizers.ts  import TabuSearch
    from saw_rl.optimizers.sa  import SimulatedAnnealing
    from saw_rl.optimizers.ga  import GeneticAlgorithm
    from saw_rl.optimizers.aco import AntColonyOptimization

    ALGOS = [
        ("TabuSearch",           lambda: TabuSearch(str(arr_path), str(dep_path), n_window=5, mps_k=3,
                                                    tabu_tenure=3, max_iters=10)),
        ("SimulatedAnnealing",   lambda: SimulatedAnnealing(str(arr_path), str(dep_path), n_window=5, mps_k=3,
                                                             t_start=100, t_min=1.0, max_iter=10)),
        ("GeneticAlgorithm",     lambda: GeneticAlgorithm(str(arr_path), str(dep_path), n_window=5, mps_k=3,
                                                           pop_size=10, n_gen=5)),
        ("AntColonyOptimization",lambda: AntColonyOptimization(str(arr_path), str(dep_path), n_window=5, mps_k=3,
                                                                n_ants=5, n_iters=5)),
    ]
    all_ok = True
    for name, factory in ALGOS:
        try:
            algo   = factory()
            result = algo.run_simulation(log_interval=0)
            assert 'total_delay_min' in result
            assert result['n_scheduled'] == algo.n_total
            ok(f"{name} — gecikme={result['total_delay_min']:.1f}dk, "
               f"süre={result['elapsed_sec']:.1f}s")
        except Exception as e:
            fail(f"{name} — {e}")
            traceback.print_exc()
            all_ok = False
    return all_ok


# ─────────────────────────────────────────────────────────────
# 6. Gerçek pipeline testi (pcap varsa)
# ─────────────────────────────────────────────────────────────
def test_real_pipeline(pcap_path: Path):
    section("6. Gerçek pipeline (pcap → CSV)")
    try:
        import pyshark  # noqa
    except ImportError:
        skip("pyshark kurulu değil — pip install pyshark")
        return True

    if not pcap_path.exists():
        skip(f"pcap bulunamadı: {pcap_path}")
        return True

    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "saw_rl.pipeline.adsb_mapper_and_parser", str(pcap_path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            ok(f"adsb_mapper_and_parser — tamamlandı")
        else:
            fail(f"adsb_mapper_and_parser hata:\n{result.stderr[:300]}")
            return False

        rapor = pcap_path.with_name(pcap_path.stem + "_rapor.txt")
        if not rapor.exists():
            fail(f"Rapor dosyası oluşmadı: {rapor}")
            return False
        ok(f"Rapor üretildi: {rapor.name}")

        result2 = subprocess.run(
            [sys.executable, "-m", "saw_rl.pipeline.adsb_preprocessor_v3", str(rapor)],
            capture_output=True, text=True
        )
        if result2.returncode == 0:
            ok("adsb_preprocessor_v3 — tamamlandı")
        else:
            fail(f"adsb_preprocessor_v3 hata:\n{result2.stderr[:300]}")
            return False

        arr = rapor.with_name(rapor.stem + "_saw_arrivals.csv")
        dep = rapor.with_name(rapor.stem + "_saw_departures.csv")
        if arr.exists() and dep.exists():
            import pandas as pd
            ok(f"arrivals.csv — {len(pd.read_csv(arr))} satır")
            ok(f"departures.csv — {len(pd.read_csv(dep))} satır")
        else:
            fail("CSV dosyaları oluşmadı")
            return False

        return True
    except Exception as e:
        fail(f"Pipeline hatası: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Ana akış
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SAW RL Ortam Doğrulama")
    parser.add_argument("--pcap", default=None,
                        help="Gerçek pcap dosyası (örn. data/kayit.pcapng)")
    args = parser.parse_args()

    print("\n" + "=" * 56)
    print("  SAW RL — Ortam Doğrulama")
    print("=" * 56)

    results = []
    results.append(test_imports())
    results.append(test_constants())

    section("3. Sentetik CSV üretimi")
    tmp = Path("/tmp/saw_rl_test")
    tmp.mkdir(exist_ok=True)
    arr_path, dep_path = make_synthetic_csvs(tmp)
    ok(f"arrivals  → {arr_path}")
    ok(f"departures → {dep_path}")

    results.append(test_runway_env(arr_path, dep_path))
    results.append(test_optimizers(arr_path, dep_path))

    if args.pcap:
        results.append(test_real_pipeline(Path(args.pcap)))

    # Özet
    passed = sum(results)
    total  = len(results)
    print(f"\n{'=' * 56}")
    if passed == total:
        print(f"  ✓ Tüm testler geçti ({passed}/{total})")
        print(f"  Benchmark için hazır:")
        print(f"  python benchmark.py --arr arrivals.csv --dep departures.csv")
    else:
        print(f"  ✗ {total - passed}/{total} test başarısız")
        print(f"  Yukarıdaki hataları düzelt, sonra benchmark'a geç.")
    print("=" * 56 + "\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
