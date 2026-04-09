# SAW Runway Sequencing — Proje Context

**Tez:** Runway Optimization Using Reinforcement Learning  
**Havalimanı:** İstanbul Sabiha Gökçen (SAW/LTFJ)  
**Son güncelleme:** 2026-04

Detaylar → `docs/architecture.md`, `docs/decisions.md`, `docs/status.md`

---

## Hızlı referans

| Kavram | Değer |
|--------|-------|
| Rolling window | N=10 |
| MPS_K | 3 |
| N_FEATURES | 10 (9 kinematik + wake sep) |
| Obs shape | (100,) |
| Separation | RECAT-EU |
| Algoritma | MaskablePPO (sb3-contrib) |
| Baseline | FCFS ~3040 dk |
| Seed | 42 |

---

## Komutlar

```bash
# Eğitim
python -m saw_rl.rl.train --arr data/adsb2_rapor_saw_arrivals.csv --dep data/adsb2_rapor_saw_departures.csv --seed 42

# Benchmark (tüm algoritmalar)
python benchmark.py --arr data/adsb2_rapor_saw_arrivals.csv --dep data/adsb2_rapor_saw_departures.csv

# Pipeline
python -m saw_rl.pipeline.adsb_mapper_and_parser kayit.pcapng
python -m saw_rl.pipeline.adsb_preprocessor_v3 kayit_rapor.txt
```

---

## Proje yapısı

```
saw_rl_pkg/
├── CLAUDE.md
├── benchmark.py
├── docs/            ← mimari, kararlar, durum
├── data/            ← CSV'ler ve sonuçlar
├── models/          ← eğitilmiş model
├── logs/            ← tensorboard
└── saw_rl/
    ├── constants.py      ← WAKE_SEP, ROT, get_sep() — TEK KAYNAK
    ├── pipeline/         ← pcap→csv
    ├── rl/               ← runway_env.py, train.py
    └── optimizers/       ← ts, sa, ga, aco
```

---

## Import

```python
from saw_rl.constants import get_sep, WAKE_SEP
from saw_rl.rl.runway_env import RunwayEnv
from saw_rl.optimizers.ts import TabuSearch
```
