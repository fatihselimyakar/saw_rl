```markdown
# Mimari

## Veri akışı

```text
piaware (Raspberry Pi)
    ↓ pcap kayıt
adsb_mapper_and_parser.py
    ↓ _rapor.txt + _rota.kml
adsb_preprocessor_v3.py
    ↓ _saw_arrivals.csv + _saw_departures.csv
    ├── real_conflict_checker.py   (kalite doğrulama)
    ├── RunwayEnv + MaskablePPO    (RL eğitimi, VecNormalize ile)
    └── benchmark.py               (tüm algoritmalar)
```

## SAW Pist Konfigürasyonu

- **06/24:** Ana iniş pisti
- **04/22:** Kalkış pisti (aynı zamanda alternate iniş)
- **Ayrım:** 1 140 m → dependent parallel operasyon
- **Staggered separation:** 45 s (çapraz pist bağımlı iniş)

## State Space (RunwayEnv)

`N_WINDOW × N_FEATURES` = 10 × 10 = 100 boyutlu gözlem vektörü.  
Her uçak için 10 feature bulunur. **Stable Baselines3 VecNormalize** ile dinamik olarak normalize edilir (`norm_obs=True`):

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
| 9 | `current_sep` | Önceki uçakla gereken ayrım (Wake + ROT) (sn) |

## Action Space

`Discrete(N_WINDOW)` — penceredeki hangi uçağın sıradaki planlanacağı.

Action masking ile MPS_K dışına çıkan seçimler yasaklanır.

## Reward Fonksiyonu (Göreceli Boşluk / Relative Gap)

Modelin amacı uçağın gecikmesini değil, **pistin boş kaldığı süreyi (gap)** minimize etmektir. Ödül, FCFS'ye kıyasla kazanılan/kaybedilen zaman farkı üzerinden hesaplanır.

```python
gap_model = (sched_ts_model - last_ts) / 60.0
gap_fcfs  = (sched_ts_fcfs - last_ts) / 60.0

reward = (gap_fcfs - gap_model) * 10.0
       [ + R_CPS_VIOLATION  (MPS ihlali varsa) ]

R_CPS_VIOLATION = -10.0
MAX_DELAY_MIN   =  90.0   (truncation eşiği)
```
*(Not: Modelin bu sinyali doğru yorumlayabilmesi için eğitim sırasında ödül normalizasyonu kapatılmıştır: `norm_reward=False`)*

## Meta-Heuristik Mimari

```text
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

Eski mimaride `_obs_lo` / `_obs_hi` array'leri ile manuel clipping uygulanıyordu (Bu konudaki eski global array hatası için bkz. ADR-003).

**Güncel Mimari:**
Manuel statik limitler büyük oranda terkedilmiş olup **Stable Baselines3 VecNormalize** wrapper'ı kullanılmaktadır.
- **`norm_obs=True`**: Gözlemlerin dinamik olarak hareketli ortalaması alınarak normalize edilir. Model bu sayede yeni veri setlerindeki farklı ETA sapmalarına karşı "kör" olmaz.
- **`norm_reward=False`**: Göreceli Ödül (Relative Gap) mantığındaki FCFS referans (sıfır) noktasının kaybolmaması için ödül normalizasyonu kapalı tutulur (bkz. ADR-014).
```