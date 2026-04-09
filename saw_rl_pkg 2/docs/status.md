# Güncel Durum

**Son güncelleme:** 2026-04-08

## Tamamlanan

### Veri pipeline
- [x] piaware ile SAW ADS-B verisi toplama
- [x] `adsb_mapper_and_parser.py` — pcap → txt + kml (tshark direct, ~10x hızlı)
- [x] `adsb_preprocessor_v3.py` — ETA/ETD, feature extraction, CSV çıktısı
- [x] `real_conflict_checker.py` — ground truth karşılaştırma, wake sep kontrolü

### RL ortamı
- [x] `RunwayEnv` — Gymnasium uyumlu environment
- [x] Rolling window (N=10), action masking, MPS_K=3 kısıtı
- [x] FCFS baseline
- [x] MaskablePPO eğitim scripti (`train.py`) — seed=42 ile tekrarlanabilir
- [x] NaT target_time satırları filtreleniyor (ADR-010)
- [x] Numpy cache ile hızlandırılmış environment (~10x)
- [x] **500k timestep eğitim tamamlandı → 3038.1 dk (%0.1 iyileşme)**

### Meta-heuristikler
- [x] Tabu Search (`ts.py`) — 2981.2 dk (%2.0 iyileşme)
- [x] Simulated Annealing (`sa.py`) — 2981.2 dk (%2.0 iyileşme)
- [x] Genetic Algorithm (`ga.py`) — 3055.9 dk (FCFS'den %0.5 kötü)
- [x] Ant Colony Optimization (`aco.py`)

### Refactor (2025-04)
- [x] `constants.py` — tek kaynak prensibi
- [x] `BaseOptimizer` — ortak altyapı
- [x] `runway_env.py` — global `_HI` mutasyon bug'ı düzeltildi
- [x] `benchmark.py` — birleşik karşılaştırma (RL dahil)

### Benchmark
- [x] `data/results.json` — tüm algoritma sonuçları (RL dahil)

---

## Devam Eden / Planlanmış

### Kısa vade
- [ ] Çok günlük veri toplama (hedef: 30 gün, ~12.000 uçak)
- [ ] Çok günlük veriyle RL eğitimi — genelleşme testi

### Orta vade (tez için kritik)
- [ ] Preprocessor v4 — trend-based ETA (scipy linregress), go-around detection
- [ ] İki pist modellemesi — `RunwayEnv`'e runway assignment action'ı ekle
- [ ] Curriculum learning — kolay → zor veri seti sıralaması

### Uzun vade (hibrit hedef)
- [ ] RL + TS hibrit — PPO karar alır, TS lokal iyileştirir
- [ ] Tez karşılaştırma tablosu (FCFS / TS / SA / GA / ACO / PPO)

---

## Benchmark Sonuçları (1 günlük veri, 420 uçak, MPS_K=3)

| Algoritma | Gecikme (dk) | Ort/uçak | İhlal | Süre |
|-----------|-------------|----------|-------|------|
| FCFS | 3040.8 | 7.24 | 0 | <1s |
| Tabu Search | 2981.2 | 7.10 | 0 | 8s |
| Simulated Annealing | 2981.2 | 7.10 | 0 | 19s |
| Genetic Algorithm | 3055.9 | 7.28 | 0 | 8s |
| **MaskablePPO (RL)** | **3038.1** | **7.23** | 1 | ~12 dk eğitim |

**Yorum:** Tek günlük veriyle RL, FCFS'e eşdeğer performans gösterdi.
TS/SA %2 iyileşme sağladı. RL'nin asıl avantajı çok günlük veriye
genelleşebilmesinde — tek çalışmada öğrenir, her seferinde arama yapmaz.

---

## Açık Sorular

1. **Çok günlük veri:** 30 gün veri toplandığında RL TS/SA'yı geçecek mi?
2. **İki pist modellemesi:** Runway assignment tezin scope'una giriyor mu?
3. **1 ihlal:** RL değerlendirmede 1 CPS ihlali yapıyor, kabul edilebilir mi?

---

## Bilinen Kısıtlamalar

- Preprocessor `hex` bazlı micro-offset hesabı kırılgan (ADR-007)
- GA, `deap` global `creator` state kullandığından thread-safe değil
- RL tek günlük veriyle FCFS'yi anlamlı ölçüde geçemiyor — veri çeşitliliği gerekiyor
