# Mimari

## Veri akışı

```
piaware (Raspberry Pi)
    ↓ pcap kayıt
adsb_mapper_and_parser.py
    ↓ _rapor.txt + _rota.kml
adsb_preprocessor_v3.py
    ↓ _saw_arrivals.csv + _saw_departures.csv
    ├── real_conflict_checker.py   (kalite doğrulama)
    ├── RunwayEnv + MaskablePPO    (RL eğitimi)
    └── benchmark.py               (tüm algoritmalar)
```

## SAW Pist Konfigürasyonu

- **06/24:** Ana iniş pisti
- **04/22:** Kalkış pisti (aynı zamanda alternate iniş)
- **Ayrım:** 1 140 m → dependent parallel operasyon
- **Staggered separation:** 45 s (çapraz pist bağımlı iniş)

## State Space (RunwayEnv)

`N_WINDOW × N_FEATURES` = 10 × 9 = 90 boyutlu gözlem vektörü.  
Her uçak için 9 feature, `[-1, 1]` aralığına normalize edilmiş:

| Index | Feature | Açıklama |
|-------|---------|----------|
| 0 | `eta_sec` | ETA - şimdiki zaman (sn) |
| 1 | `dist_km` | Piste uzaklık (km) |
| 2 | `alt_last` | Son ölçülen irtifa (ft) |
| 3 | `gs_avg` | Ortalama yer hızı (kt) |
| 4 | `baro_rate_med` | Medyan baro rate (ft/min) |
| 5 | `cat_enc` | RECAT kategorisi (0–5) |
| 6 | `phase_enc` | 0=ARRIVAL, 1=DEPARTURE |
| 7 | `fcfs_rank` | FCFS sırası |
| 8 | `hdg_diff` | Piste yöneliş farkı (°) |

## Action Space

`Discrete(N_WINDOW)` — penceredeki hangi uçağın sıradaki planlanacağı.

Action masking ile MPS_K dışına çıkan seçimler yasaklanır.

## Reward Fonksiyonu

```
reward = R_SUCCESS + R_DELAY_PER_MIN × delay_min
       [ + R_CPS_VIOLATION  (MPS ihlali varsa) ]

R_SUCCESS       = +100.0
R_DELAY_PER_MIN =  -2.0
R_CPS_VIOLATION = -100.0
MAX_DELAY_MIN   =  90.0   (truncation eşiği)
```

## Meta-Heuristik Mimari

```
BaseOptimizer (abstract)
├── _load_data()       — CSV yükleme, numpy cache
├── _eval_sequence()   — maliyet = gecikme + MPS cezası
├── run_simulation()   — kayan pencere döngüsü
└── optimize_window()  ← alt sınıf implemente eder

    ├── TabuSearch          (tabu_tenure=5, max_iters=50)
    ├── SimulatedAnnealing  (T=1000→0.1, alpha=0.90)
    ├── GeneticAlgorithm    (pop=50, gen=20, PMX crossover)
    └── AntColonyOptimization (n_ants=20, n_iters=30, elitist)
```

## Normalizasyon

`_obs_lo` / `_obs_hi` **instance-level** array'ler — her `RunwayEnv`
örneği kendi kopyasını tutar. `_obs_hi[7]` = `n_total` (veri setine göre dinamik).

> ⚠️ Eski kodda `_HI[7]` global array'i mutasyona uğratıyordu.
> Bu bug `runway_env.py` refactor'ında düzeltildi. (bkz. `docs/decisions.md` ADR-003)
