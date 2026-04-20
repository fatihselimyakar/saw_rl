# Güncel Durum

**Son güncelleme:** 2026-04-12 (hiperparametre tuning)

## Tamamlanan

### Veri pipeline
- [x] piaware ile SAW ADS-B verisi toplama
- [x] `adsb_mapper_and_parser.py` — pcap → txt + kml (tshark direct, ~10x hızlı)
- [x] `adsb_preprocessor_v3.py` — ETA/ETD, feature extraction, CSV çıktısı
- [x] `real_conflict_checker.py` — ground truth karşılaştırma, wake sep kontrolü

### RL ortamı (Optimizasyon Fazı)
- [x] `RunwayEnv` — Gymnasium uyumlu environment
- [x] Rolling window (N=10), action masking, MPS_K=3 kısıtı
- [x] FCFS baseline (3040.8 dk)
- [x] State körlüğünün giderilmesi — `VecNormalize` (Dinamik Normalizasyon)
- [x] Kelebek Etkisinin çözülmesi — Göreceli Boşluk (Relative Gap) Ödül Mimarisi
- [x] Altın Model Yakalama — Sabit Yüksek LR, Yüksek Entropi ve `EvalCallback`
- [x] **2 Milyon timestep eğitim tamamlandı** — training log min: ~3029.6 dk *(bkz. not)*
- [x] **Hibrit Reward (α=10, β=10) deneyi tamamlandı** — training log min: ~2994.8 dk *(bkz. not)*
- [x] **VecNormalize eval senkronizasyon bug fix** — `EvalVecNormSyncCallback` eklendi (ADR-017); önceki modellerin benchmark değerleri eğitim log minimumuyla karışmıştı, yeniden eğitim gerekiyor
- [x] **Hiperparametre tuning** — LR linear schedule (3e-4→5e-5), ent_coef=0.1, n_steps=4096 (ADR-019); 500k adımda FCFS geçildi: 3024.9 dk

### Meta-heuristikler
- [x] Tabu Search (`ts.py`) — 3001.6 dk (çalıştırmalar arası ~20 dk varyans normal)
- [x] Simulated Annealing (`sa.py`) — 3001.6 dk
- [x] Genetic Algorithm (`ga.py`) — 3055.9 dk
- [x] Ant Colony Optimization (`aco.py`)

### Refactor
- [x] `constants.py` — tek kaynak prensibi
- [x] `BaseOptimizer` — ortak altyapı
- [x] `benchmark.py` — birleşik karşılaştırma (RL için `VecNormalize` destekli)
- [x] `benchmark.py` iyileştirmeleri — `--model_dir` parametresi, GA/ACO lazy import (ADR-018)
- [x] `evaluate.py` — Eğitilmiş en iyi model için bağımsız test betiği

---

## Devam Eden / Planlanmış

### Kısa vade
- [x] **Hibrit Reward mimarisi** — `(gap_fcfs - gap_model) * α - (model_delay / MAX_DELAY) * β` formülü tamamlandı (bkz. ADR-016)
- [ ] **2M adım tam eğitim** — LR schedule + ent_coef=0.1 + n_steps=4096 ile (ADR-019 config); 500k'da FCFS geçildi, 2M'de TS/SA seviyesi hedefleniyor
- [ ] Çok günlük veri toplama (hedef: 30 gün, ~12.000 uçak)
- [ ] Çok günlük veriyle RL eğitimi — RL'in asıl "genelleştirme" gücünü kanıtlama

### Orta vade (tez için kritik)
- [ ] **İki pist modellemesi** — `RunwayEnv`'e runway assignment action'ı ekle (öncelikli)
- [ ] Preprocessor v4 — trend-based ETA (scipy linregress), go-around detection
- [ ] Curriculum learning — kolay → zor veri seti sıralaması

### Uzun vade (hibrit hedef)
- [ ] RL + TS hibrit — PPO hızlı karar alır, TS (lokal arama ile) son dokunuşu yapar.

---

## Benchmark Sonuçları (1 günlük veri, 420 uçak, MPS_K=3)

| Algoritma | Gecikme (dk) | Ort/uçak | İhlal | Süre |
|-----------|-------------|----------|-------|------|
| FCFS | 3040.8 | 7.24 | 0 | <1s |
| Tabu Search | 3001.6 | 7.15 | 0 | 8s |
| Simulated Annealing | 3001.6 | 7.15 | 0 | 20s |
| Genetic Algorithm | 3055.9 | 7.28 | 0 | 8s |
| MaskablePPO (500k, ADR-019 config) | **3024.9** | **7.22** | 0 | — |

> **Not (ADR-017/019):** Önceki RL sonuçları (3029.6, 2994.8 dk) stochastic training minimumlarıydı. ADR-017 fix + ADR-019 hiperparametre tuning sonrası 500k adımda deterministic eval **3024.9 dk** elde edildi — FCFS geçildi. 2M adım eğitimde TS/SA seviyesi (~3001 dk) hedefleniyor.

**TS/SA notu:** Önceki çalıştırmada 2981.2 dk; şimdiki çalıştırmada 3001.6 dk. Her iki algoritma da deterministik değil (SA sıcaklık, TS komşu seçimi); çalıştırmalar arası ~20 dk varyans normal.

---

## Açık Sorular

1. **Büyük Veri Avantajı:** 30 günlük veriye geçildiğinde, çok çeşitli trafik paternlerini gören RL, TS/SA'yı tamamen otonom olarak geçebilecek mi?
2. **İki pist modellemesi:** Runway assignment tezin scope'una giriyor mu? (öncelikli hedef olarak belirlendi)

---

## Bilinen Kısıtlamalar

- Preprocessor `hex` bazlı micro-offset hesabı kırılgan (ADR-007)
- GA, `deap` global `creator` state kullandığından thread-safe değil
- TS/SA çalıştırmalar arası ~20 dk varyans gösteriyor (deterministik değil)