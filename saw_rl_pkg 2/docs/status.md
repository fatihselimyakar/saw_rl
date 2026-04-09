# Güncel Durum

**Son güncelleme:** 2026-04-10

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
- [x] **2 Milyon timestep eğitim tamamlandı → 3029.6 dk (FCFS'ten 11 dakika daha iyi, %0.4 iyileşme, SIFIR MPS İhlali)**

### Meta-heuristikler
- [x] Tabu Search (`ts.py`) — 2981.2 dk 
- [x] Simulated Annealing (`sa.py`) — 2981.2 dk
- [x] Genetic Algorithm (`ga.py`) — 3055.9 dk
- [x] Ant Colony Optimization (`aco.py`)

### Refactor
- [x] `constants.py` — tek kaynak prensibi
- [x] `BaseOptimizer` — ortak altyapı
- [x] `benchmark.py` — birleşik karşılaştırma (RL için `VecNormalize` destekli)
- [x] `evaluate.py` — Eğitilmiş en iyi model için bağımsız test betiği

---

## Devam Eden / Planlanmış

### Kısa vade
- [ ] Çok günlük veri toplama (hedef: 30 gün, ~12.000 uçak)
- [ ] Çok günlük veriyle RL eğitimi — RL'in asıl "genelleştirme" gücünü kanıtlama

### Orta vade (tez için kritik)
- [ ] Preprocessor v4 — trend-based ETA (scipy linregress), go-around detection
- [ ] İki pist modellemesi — `RunwayEnv`'e runway assignment action'ı ekle
- [ ] Curriculum learning — kolay → zor veri seti sıralaması

### Uzun vade (hibrit hedef)
- [ ] RL + TS hibrit — PPO hızlı karar alır, TS (lokal arama ile) son dokunuşu yapar.

---

## Benchmark Sonuçları (1 günlük veri, 420 uçak, MPS_K=3)

| Algoritma | Gecikme (dk) | Ort/uçak | İhlal | Süre |
|-----------|-------------|----------|-------|------|
| FCFS | 3040.8 | 7.24 | 0 | <1s |
| Tabu Search | 2981.2 | 7.10 | 0 | 8s |
| Simulated Annealing | 2981.2 | 7.10 | 0 | 19s |
| Genetic Algorithm | 3055.9 | 7.28 | 0 | 8s |
| **MaskablePPO (RL - Best Model)** | **3029.6** | **7.21** | **0** | **<2s** (Inference) |

**Yorum:** RL, otonom olarak ve sezgisel bir arama (heuristic search) yapmadan, FCFS baseline'ını net bir şekilde kırarak (3040 -> 3029 dk) Wake Turbulence ayrımlarını fırsata çevirmeyi öğrenmiştir. Üstelik sıfır kural ihlali yapmıştır. 

TS/SA hala %2'lik (2981 dk) bir üstünlüğe sahip olsa da, bu algoritmalar her yeni günde saniyeler/dakikalar boyunca hesaplama yapmak zorundadır. Eğitilmiş RL ajanı ise yepyeni bir duruma (Inference) milisaniyeler içinde cevap verebilir.

---

## Açık Sorular

1. **Büyük Veri Avantajı:** 30 günlük veriye geçildiğinde, çok çeşitli trafik paternlerini gören RL, TS/SA'yı tamamen otonom olarak geçebilecek mi?
2. **İki pist modellemesi:** Runway assignment tezin scope'una giriyor mu?

---

## Bilinen Kısıtlamalar

- Preprocessor `hex` bazlı micro-offset hesabı kırılgan (ADR-007)
- GA, `deap` global `creator` state kullandığından thread-safe değil