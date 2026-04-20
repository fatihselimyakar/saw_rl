# SAW Runway Sequencing — Sunum İçeriği
**Süre:** ~30 dakika | **Hedef kitle:** RL uzmanı akademisyen

---

## Slayt 1 — Başlık

# Runway Optimization Using Reinforcement Learning
### İstanbul Sabiha Gökçen Havalimanı (SAW/LTFJ)

**MaskablePPO ile Tek Pist Sıralama Optimizasyonu**

---

## Slayt 2 — Problem Tanımı

### Problem: Pist Sıralama Optimizasyonu

- SAW'da günde ~420 uçak hareketi, **tek pist**
- **İniş ve kalkış aynı pistı paylaşıyor** — sıralama her ikisini birden kapsıyor
- Mevcut yaklaşım: **FCFS** — optimizasyon yok, 3040.8 dk toplam gecikme
- Ardışık uçaklar arasında wake turbulence separasyonu zorunlu (güvenlik)
- Pist gereksiz boş kalıyor, uçaklar gereğinden fazla bekliyor

**Ne yapmak istiyoruz?**
- Sıralama kararını bir RL problemi olarak modelle
- Ajan her adımda bir uçak seçiyor, piste gönderiyor
- Güvenlik kurallarını ihlal etme, toplam gecikmeyi minimize et
- FCFS'ten daha iyi bir sıralama politikası öğren

**Kısıtlar:**
- **RECAT-EU** wake turbulence kategorileri — ihlal edilemez (güvenlik)
- **MPS_K = 3** — bir uçak FCFS sırasından en fazla ±3 pozisyon kaydırılabilir
  *(ICAO/Eurocontrol DMAN standardı — pilot yakıt planı, bağlantı uçuşları, slot kısıtları)*
- **Rolling window N=10** — ajan aynı anda sadece 10 uçağa bakabilir (ATC gerçekçiliği)

---

## Slayt 3 — Veri Pipeline

### Gerçek ADS-B Verisi: Uçtan Uca Pipeline

**1. piaware (SDR alıcı) → .pcapng**
```
Ham 1090 MHz ADS-B yayınları, ağ paketi olarak kaydedildi
```

**2. tshark → rapor.txt**
```
KAYIT #1 | ZAMAN: 2026-02-17 16:44:04
flight    : AMQ2562
category  : A3          ← RECAT-EU kategori
alt_baro  : 34000 ft
hex       : 45213e
rssi      : -16.8
```

**3. adsb_preprocessor_v3.py → arrivals.csv**
```
flight   category  target_time           dist_km   gs_avg   baro_rate
TCIHA    A2        2026-02-17 21:25:40   48.23     370.3    -2976.0
MSR542   A3        2026-02-18 00:12:53    9.26     272.9    -1664.0
MSR735   A3        2026-02-17 17:41:39   17.51     373.3    -1152.0
```

**3b. adsb_preprocessor_v3.py — Feature Extraction**

Ham ADS-B sinyalinden iki tür özellik üretilir:

*Doğrudan ADS-B'den okunan:*
| Feature | Açıklama |
|---------|----------|
| `alt_last` | Son ölçülen irtifa (ft) |
| `gs_avg` | Son 10 ölçümün medyanı, [120–600 kt] kırpılmış |
| `baro_rate_med` | Barometrik alçalma/tırmanma hızı (ft/dk) |
| `category` | RECAT-EU kategori etiketi (A1–A5) |
| `rssi_last` | Son sinyal gücü (dBFS) |
| `nac_p_last` | Navigasyon hassasiyet kodu |

*Hesaplanan (türetilen):*
| Feature | Nasıl? |
|---------|--------|
| `dist_km` | Haversine formülü — son pozisyon → pist eşiği |
| `bearing` | Son pozisyon ile pist yönü arasındaki açı |
| `hdg_diff` | Uçak başlığı ile pist yönü arasındaki açı farkı |
| `phase` | İniş / kalkış / cruise — irtifa + baro_rate'e göre |
| `target_time` (ETA/ETD) | `max(t_horizontal, t_vertical)` — mesafe/hız ve irtifa/alçalma hızından |

**4. RunwayEnv → observation vektörü**
```python
# Her uçak için 10 feature → 10 uçak → obs shape (100,)
[dist_km, alt, gs, hdg_diff, baro_rate, cat_norm, rssi, n_total, wake_sep, delay]
```

**Veri boyutu:** 1 günlük trafik, **420 uçak** (iniş + kalkış)

---

## Slayt 4 — Ortam Tasarımı: RunwayEnv

### Gymnasium Uyumlu RL Ortamı

---

**Observation Space: (100,) = 10 uçak × 10 feature**

Her uçak için şu 10 özellik hesaplanır, [-1, 1] aralığına normalize edilir:

| # | Feature | Açıklama |
|---|---------|----------|
| 0 | `eta_delta` | Şu andan uçağın ETA'sına kadar kalan süre (sn) |
| 1 | `dist_km` | Piste kalan mesafe (km) |
| 2 | `alt_last` | Son irtifa (ft) |
| 3 | `gs_avg` | Ortalama yer hızı (knot) |
| 4 | `baro_rate` | İniş/çıkış hızı (ft/dk) |
| 5 | `cat_enc` | RECAT-EU kategorisi (A1–A5 → 1–5) |
| 6 | `phase_enc` | İniş=0, Kalkış=1 |
| 7 | `fcfs_rank` | Uçağın orijinal FCFS sıra numarası |
| 8 | `hdg_diff` | Uçağın başlığı ile pist yönü açı farkı (°) |
| 9 | `wake_sep` | Bir önceki uçakla gereken wake separasyonu (sn) |

---

**Action Space: Discrete(10)**

Her adımda penceredeki 10 uçaktan biri seçilir → o uçak piste gönderilir.

**Action Masking — MPS_K=3:**
```python
def action_masks(self):
    for i, idx in enumerate(window):
        if abs(sched_pos - fcfs_rank[idx]) <= self.mps_k:
            mask[i] = True   # geçerli seçim
        # mask[i] = False → softmax'tan önce sıfırlanır
```
- Ajan MPS_K=3'ü ihlal eden uçağı hiç seçemez
- Geçersiz action için ceza öğrenmek gerekmez → daha hızlı ve kararlı öğrenme

---

**Episode Akışı — Tek Adımda Ne Oluyor?**

```
1. _get_window()     → ETA'ya göre sıralı ilk 10 uçak seçilir
2. action_masks()    → MPS_K ihlali yapacak seçimler maskelenir
3. model.predict()   → agent maskeli action space üzerinden seçim yapar
4. step(action)      → seçilen uçak piste gönderilir:
     - wake sep kontrolü: sched_ts = max(ETA, last_ts + sep)
     - gap_model ve gap_fcfs hesaplanır
     - reward üretilir
     - done_mask güncellenir, pencere bir ilerler
5. 420 adım sonra episode biter
```

**Örnek adım:**
```
Adım 87 | 14:23:10 | Kalan: 333 uçak | Gecikme: 142.3 dk
  [0] A3 ARR  ETA:14:23:45  dist:8km   ✓  ← seçilebilir (FCFS rank farkı ≤ 3)
  [1] A2 DEP  ETA:14:24:12  dist:2km   ✓
  [2] A4 ARR  ETA:14:25:01  dist:15km  ✗  ← maskeli (FCFS rank farkı > 3)
  ...
```

---

## Slayt 5 — Reward Fonksiyonu: Geliştirme Süreci

### Üç Deneme, İki Başarısızlık

---

**Deneme 1 — Mutlak Gecikme Cezası**
```python
reward = -total_delay
```
**Sorun: Kelebek Etkisi**
Erken adımlardaki küçük hatalar zincirleme büyüyerek son adımlarda
devasa cezalara dönüştü. PPO value function çöktü, öğrenme durdu.

---

**Deneme 2 — FCFS Karşılaştırması**
```python
reward = delay_fcfs - delay_model
```
**Sorun: Reward Hacking**
Model geç gelen uçakları (zaten `delay=0`) seçmeye başladı.
Teknik olarak reward yüksek, ama toplam gecikme **4025 dk'ya** çıktı.

---

**Final — Hibrit Göreceli Ödül**
```python
reward = (gap_fcfs - gap_model) × α  −  (model_delay / MAX_DELAY) × β
# α=10, β=10
```

| Terim | Ne ölçüyor? | Neden hile yapılamaz? |
|-------|------------|----------------------|
| `gap_fcfs - gap_model` | Pist verimliliği | FCFS referans sabit |
| `model_delay / MAX_DELAY` | Bireysel gecikme | Geç gelen uçağın delay=0'dır |

**FCFS terimi matematiksel notu:**

`gap_fcfs` her adımda sabit bir referans (model onu kontrol etmiyor).
Yani gradyan açısından `(gap_fcfs - gap_model) × α ≡ -gap_model × α + sabit`.
FCFS terimi öğrenmeyi değiştirmiyor; **sıfır referans noktası** olarak kaldı:
- `reward > 0` → model bu adımda FCFS'ten daha verimli pist kullandı
- `reward < 0` → bu adımda FCFS daha iyiydi
Sayısal katkısı yok, yorumlanabilirlik ve eğitim kararlılığı için tutuldu.

---

## Slayt 6 — Eğitim Altyapısı: train.py

### MaskablePPO + VecNormalize

**Algoritma:** `MaskablePPO` (sb3-contrib)
- Standart PPO + action masking desteği

**Normalizasyon:** `VecNormalize`
- Dinamik running mean/var ile obs normalizasyonu
- `norm_reward=False` — reward sinyali korunur (ADR-014)

---

### Kritik Bug — ADR-017: EvalCallback VecNormalize Desenkronizasyonu

**Sorun:**
```
training env  →  obs_rms güncelleniyor (mean≠0, var≠1)
eval env      →  default istatistikler (mean=0, var=1) — hiç güncellenmiyor
```
Model eğitimde normalized obs görürken, eval'de ham obs görüyordu.
"En iyi model" tamamen yanlış metriğe göre seçiliyordu.

**Fix: `EvalVecNormSyncCallback`**
```python
def _on_step(self) -> bool:
    if self.n_calls % self.eval_freq == 0:
        self.eval_vn.obs_rms = copy.deepcopy(self.train_vn.obs_rms)
    return True
```
Her eval öncesi training istatistikleri eval env'e kopyalanır.

**Sonuç:** Bug öncesi raporlanan "iyi" sonuçlar (2994.8 dk) stochastic training
episode minimumlarıydı — gerçek deterministic eval değil.

---

## Slayt 7 — Hiperparametre Tuning

### Aşamalı İyileştirme Süreci

| Konfigürasyon | En iyi eval reward | Deterministik gecikme | Gözlem |
|--------------|-------------------|----------------------|--------|
| Sabit LR=3e-4, ent=0.05, n_steps=2048 | -433.84 @ 120k | 3305.4 dk | Collapse @ 130k |
| LR schedule 3e-4→5e-5, ent=0.05 | -429.03 @ 280k | 3305.4 dk | Collapse @ 340k'ya ertelendi |
| LR schedule, ent=0.1 | -417.69 @ 480k | — | Collapse yok, 500k boyunca iyileşme |
| **LR schedule, ent=0.1, n_steps=4096** | **-364.17 @ 420k** | **3024.9 dk** | **FCFS geçildi ✓** |

**n_steps=4096'nın etkisi:**
- Her güncellemede ~10 episode (~4096 adım) veri toplanıyor (önceden ~5 episode)
- Daha iyi credit assignment → model hangi kararın gecikmeyi azalttığını daha iyi öğreniyor

**Final hiperparametreler:**
```python
learning_rate = LinearSchedule(3e-4, 5e-5, 1.0)
ent_coef      = 0.1
n_steps       = 4096
batch_size    = 256
n_epochs      = 10
net_arch      = [256, 256, 256]
gamma         = 0.99,  gae_lambda = 0.95
```

---

## Slayt 8 — Sonuçlar

### Benchmark: 1 Günlük Veri, 420 Uçak, MPS_K=3

| Algoritma | Toplam Gecikme | Ort/Uçak | Separasyon İhlali |
|-----------|---------------|----------|-------------------|
| FCFS (baseline) | 3040.8 dk | 7.24 dk | 0 |
| Genetic Algorithm | 3055.9 dk | 7.28 dk | 0 |
| **MaskablePPO (500k adım)** | **3024.9 dk** | **7.22 dk** | **0** |
| Tabu Search | 3001.6 dk | 7.15 dk | 0 |
| Simulated Annealing | 3001.6 dk | 7.15 dk | 0 |

- RL **FCFS'i geçti** (−15.9 dk, %0.5 iyileştirme)
- TS/SA ile aradaki fark: 23.3 dk — 2M adım eğitimde kapanması bekleniyor
- Tüm yöntemler separasyon ihlali sıfır

---

## Slayt 9 — Meta-Heuristikler (Referans Noktaları)

### Karşılaştırma Algoritmaları

Aynı ortam (`BaseOptimizer`), aynı kısıtlar (MPS_K=3, RECAT-EU):

- **Tabu Search** — yasaklı hamle listesiyle lokal aramadan kaçınma
- **Simulated Annealing** — sıcaklık parametresiyle kötü hamleler kabul edilebilir
- **Genetic Algorithm** — çaprazlama ve mutasyon ile popülasyon tabanlı arama
- **Ant Colony Optimization** — feromon birikimiyle olasılıksal yol oluşturma

`benchmark.py` — tüm algoritmaları tek komutla karşılaştıran merkezi script

**TS/SA ~3001 dk:** Bu problemde MPS_K=3 kısıtı altında erişilebilir alt sınırı gösteriyor.

---

## Slayt 10 — Tartışma

### Neden RL Bu Kadar Zor?

**1. Tek veri seti sorunu**
Model hep aynı 420 uçakla eğitiliyor. Farklı trafik paternlerini görmüyor.
→ Çok günlük veri ile genelleştirme gücü test edilmeli.

**2. MPS_K=3 teorik tavan**
TS/SA da ~3001 dk'da duruyor — bu kısıt altında matematiksel alt sınır orada.
RL'in önünde çok az alan var.

**3. Stochastic vs Deterministic**
Training sırasında stochastic exploration ile 3020 dk buluyor,
ama deterministic politikaya dönüştürmek zor.
ADR-017 bu farkı derinleştiriyordu.

**4. Sample efficiency**
500k adım = 500k/418 ≈ **1196 tam episode**.
Meta-heuristikler aynı sorunu çok daha az iterasyonla çözüyor.
RL'in avantajı genelleştirme — tek veri setinde değil.

---

## Slayt 11 — Sonraki Adımlar

### Yol Haritası

**Kısa vade:**
- [ ] 2M adım tam eğitim — mevcut hiperparametrelerle, TS/SA seviyesi hedef
- [ ] Birden fazla seed ile tekrarlanabilirlik testi

**Orta vade:**
- [ ] **İki pist modellemesi** — `RunwayEnv`'e runway assignment action'ı eklenir
  - Obs'a ikinci pistın doluluk durumu eklenir
  - Action space genişler: "hangi uçak" + "hangi pist"
  - RL'in meta-heuristiklere göre asıl avantaj alanı burası
- [ ] 30 günlük veri — genelleştirme ve transfer learning

**Uzun vade:**
- [ ] RL + TS hibrit — PPO hızlı karar, TS lokal arama ile iyileştirme

---

## Slayt 12 — Özet

### Katkılar

1. **Gerçek ADS-B verisi ile uçtan uca pipeline** — piaware → RL ortamı
2. **Action masking** ile geçersiz kararlar önlendi — daha kararlı öğrenme
3. **Hibrit reward mimarisi** — pist verimliliği + bireysel gecikme dengesi
4. **ADR-017 bug fix** — eval güvenilirliği sağlandı, gerçek sonuçlar ölçülebildi
5. **Hiperparametre tuning** — 500k adımda FCFS geçildi: **3024.9 dk**

### Sonuç

> MaskablePPO, gerçek ADS-B verisi üzerinde 500k adım eğitimle FCFS baseline'ını geçti.
> Meta-heuristiklerle aradaki fark 23 dk — 2M adım ve çok günlük veriyle kapanması bekleniyor.
> İki pistli modellemeye geçiş, RL'in asıl genelleştirme gücünü ortaya koyacak.
