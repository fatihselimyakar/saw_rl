# Architecture Decision Records (ADR)

Her önemli teknik karar bu dosyada belgelenir.  
Format: **Karar → Neden → Sonuç / Trade-off**

---

## ADR-001: MaskablePPO seçimi

**Karar:** `sb3-contrib` kütüphanesinden `MaskablePPO`  
**Neden:** - Runway sequencing'de bazı action'lar domain constraint'leri ihlal eder
  (MPS_K sınırı dışı uçak seçimi). Standart PPO bu action'ları ceza ile
  öğrenmeye çalışır → yavaş öğrenme, kararsız politika.
- MaskablePPO geçersiz action'ları softmax'tan önce maskeleyerek
  agent'ın hiç denemesini engeller → daha hızlı ve güvenli öğrenme.

**Trade-off:** `sb3-contrib` bağımlılığı eklendi, `ActionMasker` wrapper gerekiyor.

---

## ADR-002: Rolling Window (N=10) yaklaşımı

**Karar:** Tüm sıralamayı değil, ETA'ya göre sıralı ilk 10 uçağı gözlemle  
**Neden:** - ATC gerçekte de yalnızca yakın horizon'daki uçaklara odaklanır.
- Tam sıralama problemi NP-hard; N=10 pencere hem sezgisel olarak doğru
  hem de yönetilebilir action space sağlıyor.
- Kayan ufuk (receding horizon) — her adımda pencere kayar, seçilen uçak listeden çıkar.

**Trade-off:** Global optimal yerine local optimal — kabul edilebilir, ATC pratiğiyle uyumlu.

---

## ADR-003: Global `_HI` mutasyonu bug fix

**Karar:** `runway_env.py`'de `_HI[7] = float(self.n_total)` satırı
instance-level `self._obs_hi[7]` olarak taşındı  
**Neden:** - Eski kod modül-level `_HI` array'ini her `RunwayEnv.__init__` çağrısında
  mutasyona uğratıyordu.
- Eğitim (`train.py`) ve değerlendirme (`evaluate_model`) farklı veri setleriyle
  arka arkaya env açarsa, normalizasyon üst sınırı son açılan env'in `n_total`'ına
  göre ayarlanıyor, önceki env'in gözlemleri yanlış normalize ediliyordu.

**Çözüm:** Her instance `OBS_LOW/HIGH`'ın kopyasını alıyor (`np.array(OBS_LOW)`),
`_obs_hi[7]`'yi kendi `n_total`'ına göre set ediyor.

---

## ADR-004: `constants.py` tek kaynak prensibi

**Karar:** `WAKE_SEP`, `ROT`, `CAT_MAP`, `get_sep()`, `normalize_cat()` tüm dosyalarda
tekrarlanmak yerine `constants.py`'de tanımlandı, diğerleri import ediyor  
**Neden:** - Orijinal kodda bu yapılar 5 dosyada (`runway_env`, `ts`, `sa`, `ga`, `aco`) ayrı ayrı
  tanımlıydı → bir ROT değeri değişince 5 yeri güncellemek gerekiyordu.
- `normalize_cat()` eklenerek `"A3MEDIUM"` gibi gürültülü ADS-B kategori string'leri
  tek bir yerde doğru şekilde parse ediliyor.

---

## ADR-005: `BaseOptimizer` abstract class

**Karar:** TS, SA, GA, ACO ortak `_load_data`, `_eval_sequence`, `run_simulation`
mantığını `BaseOptimizer`'dan miras alıyor; yalnızca `optimize_window()` override ediyor  
**Neden:** - Dört dosyada identik `run_simulation` döngüsü → debug/iyileştirme her dosyada ayrı.
- Tez için ileride hibrit algoritma (örn. RL+TS) eklenecek → tek metod yeterli olacak.

**Sonuç:** Her algoritma dosyası ~100 satırın altına düştü.

---

## ADR-006: `benchmark.py` merkezi karşılaştırma

**Karar:** Ayrı ayrı çalıştırmak yerine tüm algoritmaları tek komutla çalıştıran script  
**Neden:** - Tez için FCFS / TS / SA / GA / ACO / RL sonuçları yan yana tabloda gösterilmeli.
- `--out results.json` ile sonuçlar dışa aktarılabilir → LaTeX tablo üretimi kolaylaşır.
- `--skip` parametresiyle yavaş algoritmalar (GA, ACO) geliştirme sırasında atlanabilir.

---

## ADR-007: adsb_mapper_and_parser — pyshark → tshark direct

**Karar:** `pyshark` kütüphanesi kaldırıldı, `tshark` direkt `subprocess.Popen` ile çağrılıyor  
**Neden:** - pyshark her paketi tshark'ın XML çıktısından Python'da parse ediyor — paket başına büyük overhead.
- tshark filtrelemeyi (`http.file_data contains "aircraft"`) C'de yapıyor, Python'a sadece
  iki alan dönüyor: `frame.time_epoch` ve `http.file_data` (hex encoded).
- 10 dakika olan işlem süresi önemli ölçüde düştü.

**Teknik detay:** tshark çıktısındaki hex kolon formatı (`7b:22:6e...`) `bytes.fromhex(s.replace(':', ''))` ile
decode ediliyor. Streaming (`Popen` + satır satır okuma) sayesinde belleğe sığmayan büyük
pcap dosyaları da sorunsuz işleniyor.

**Gereksinim:** `tshark` sistem PATH'inde olmalı (pyshark zaten bunu gerektiriyordu).

---

## ADR-008: Preprocessor v3 — mevcut durum

**Karar:** `adsb_preprocessor_v3.py` şimdilik sabit tutuldu  
**Neden:** - ETA hesabı `max(t_horizontal, t_vertical)` — basit ama işlevsel.
- v4 için planlanan iyileştirmeler: scipy linregress trend-based ETA,
  go-around detection, category-based ROT farklılaştırması.
- Bu iyileştirmeler RL eğitiminden bağımsız, ayrı bir iterasyonda ele alınacak.

**Açık sorular:**
- v3 ETA hataları RL training kalitesini ne kadar etkiliyor?
  `real_conflict_checker.py` ile ölçülmeli.

---

## ADR-009: ACO feromon deposit bug fix

**Karar:** `deposit = Q / (best_fit + 1)` → `deposit = Q / (1 + best_delay_cezasiz)`  
**Neden:** İlk benchmark sonuçlarında ACO +204% gecikme ve 6 MPS ihlali üretiyordu.  
İki ayrı bug vardı:

1. **Heuristic:** `eta = 1 / (penalty + delay)` — ceza 10.000 olduğunda tüm
   infeasible seçenekler için `eta ≈ 1e-5`. Aralarındaki gecikme farkı yok oldu,
   algoritma blind seçim yapıyordu. **Fix:** `eta = 1/delay * 1e-4` çarpanı —
   infeasible caydırılıyor ama gecikme farkı korunuyor.

2. **Deposit:** `best_fit` 10.000+ olduğunda `deposit = 100/10001 ≈ 0.01`.
   Feromon birikemedi, iterasyonlar arası öğrenme gerçekleşmedi.
   **Fix:** Karınca inşası sırasında cezasız `ant_delay` ayrıca takip ediliyor;
   deposit bu değer üzerinden hesaplanıyor.

---

## ADR-010: NaT target_time filtrelemesi

**Karar:** `_load_data` içinde `dropna(subset=['target_time'])` eklendi  
**Neden:** - Arrivals CSV'sinde 10 satırda `target_time` NaT (eksik değer) bulunuyordu.
- Eski kod bu satırlara `first_ts` (veri setinin ilk zaman damgası) atıyordu.
- NaT satırları fcfs_rank ~400 civarında ama timestamp olarak en başa
  yerleşince `_get_window()` bunları hep öne alıyordu → her adımda CPS ihlali.
- 420 uçaktan 10 tanesi filtrelenerek 410 uçakla devam edildi.

**Sonuç:** CPS ihlalleri FCFS baseline'da 416'dan 0'a düştü.

---

## ADR-011: Seed ile tekrarlanabilirlik

**Karar:** `train.py`'e `set_seed(seed)` fonksiyonu eklendi, `MaskablePPO`'ya
`seed=args.seed` parametresi geçildi. Varsayılan seed=42.  
**Neden:** - RL sonuçları her çalıştırmada farklı çıkıyordu — reproducibility yok.
- Tez için belirli bir sonucu raporlamak ve tekrar üretebilmek gerekiyor.
- `random`, `numpy`, `torch` seed'leri aynı anda set ediliyor.

**Kullanım:** `python -m saw_rl.rl.train --seed 42`

---

## ADR-012: n_steps=2048 rollout boyutu

**Karar:** PPO rollout boyutu `n_steps=2048`  
**Neden:** - İlk denemede `n_steps=420` (≈1 episode) kullanıldı. Düşük varyans ama yavaş convergence.
- `n_steps=2048` (~5 episode) ile daha kararlı gradient tahminleri elde edildi.
- `batch_size=256`, `n_epochs=10` ile birlikte kullanıldığında 2M adımda kararlı öğrenme sağlandı.

**Eğitim parametreleri (güncel — bkz. ADR-015):**
```python
n_steps=2048, batch_size=256, n_epochs=10
gamma=0.99,   gae_lambda=0.95, ent_coef=0.05
net_arch=[256, 256, 256], learning_rate=3e-4  # sabit (linear schedule değil)
```

---

## ADR-013: Göreceli Boşluk (Relative Gap) Ödül Mekanizması

**Karar:** Modelin mutlak gecikme yerine "FCFS'in yaratacağı pist boşluğu ile Kendi yarattığı pist boşluğu arasındaki fark" (`(gap_fcfs - gap_model) * 10.0`) üzerinden ödüllendirilmesi.  
**Neden:** - Mutlak veya kümülatif gecikme cezalandırıldığında "Kelebek Etkisi" oluşuyor; erken adımlarda yapılan küçük hatalar sistemin sonlarında devasa cezalara dönüşerek PPO değer fonksiyonunu çökertiyordu.
- FCFS stratejisini "Sıfır" referans noktası alarak, modelin en kötü ihtimalle FCFS sırasını takip etmesi, boşluk (Wake Separation) bulduğunda ise ondan dakika çalarak pozitif puan alması hedeflendi.

---

## ADR-014: VecNormalize Geçişi ve Ödül Normalizasyonunun İptali

**Karar:** `RunwayEnv` içindeki statik gözlem limitleri terk edilip `VecNormalize` kullanılması. Ancak `norm_reward=False` olarak ayarlanması.  
**Neden:** - Statik limitler, verideki sapmalar (outliers) nedeniyle modelin "kör" olmasına ve politikanın donmasına neden oluyordu. `VecNormalize(norm_obs=True)` bunu çözdü.
- **Kritik Hata:** `norm_reward=True` yapıldığında, negatif ödüllerin ortalaması alındığı için model FCFS referans noktasını kaybetti (Catastrophic Forgetting). Ödül normalizasyonu kapatılarak modelin gerçek sinyali duyması sağlandı.

---

## ADR-015: Sabit Öğrenme Oranı, Yüksek Entropi ve EvalCallback

**Karar:** Linear schedule yerine sabit `3e-4` LR, `ent_coef=0.05` (yüksek entropi) ve `EvalCallback` kullanımı.  
**Neden:** - Zamanla azalan LR ve düşük entropi, modelin erken safhalarda bulduğu zayıf stratejilere takılıp kalmasına sebep oldu.
- Model sürekli yeni yollar denemeye (dalgalanmaya) zorlandı. Bu süreçte şans eseri veya keşif sonucu FCFS'i yendiği o "Kusursuz Sıralama" anları, `EvalCallback` sayesinde otomatik olarak test edilip `best_model.zip` adıyla kaydedildi.

---

## ADR-016: Hibrit Göreceli Ödül Mimarisi

**Karar:** Ödül fonksiyonu salt gap minimizasyonundan (`(gap_fcfs - gap_model) * α`) hibrit formüle dönüştürüldü:
```
reward = (gap_fcfs - gap_model) * α  -  (model_delay / MAX_DELAY) * β
```
`α=10, β=10` ile 2M adım eğitim tamamlandı. Training log minimum ~2994.8 dk gösterdi; ancak deterministic best_model eval'i ADR-017 bug'ı nedeniyle geçersizdi. Fix sonrası yeniden eğitim gerekiyor.

**Neden:**  
- Salt gap ödülü bazı senaryolarda yanlış optimizasyon yapıyordu. Örnek: Heavy ardından Light sıralamasında gap küçük ama gecikme büyük; Light ardından Heavy sıralamasında gap büyük ama toplam gecikme küçük. Model gap'i minimize ederek yanlış uçağı seçiyordu.  
- İlk hibrit deneme (`delay_fcfs - delay_model` karşılaştırması) reward hacking'e yol açtı: model doğal geç gelen uçakları seçerek `delay_model=0` kazanıyordu, delay 4025 dk'ya çıktı.  
- **Çözüm:** FCFS karşılaştırması kaldırıldı. `-(model_delay / MAX_DELAY) * β` formülü direkt "bu uçağa ne kadar beklettin?" sorusunu sorar — hile yapılamaz, çünkü geç gelen uçağın zaten `delay=0`'dır.

**Sonuç:** Gap sinyali pist verimliliğini, delay sinyali bireysel adalet/gecikmeyi optimize eder; ikisi birbirini tamamlar.

---

## ADR-017: EvalCallback VecNormalize Senkronizasyon Bug Fix

**Karar:** `train.py`'e `EvalVecNormSyncCallback` eklendi. Her `eval_freq` adımda training VecNormalize `obs_rms`'sini eval env'e kopyalar.

**Neden — Bug:**  
- `eval_vec_env`, `VecNormalize(training=False)` ile oluşturuluyordu → default istatistikler (mean=0, var=1), hiçbir zaman güncellenmiyordu.  
- Model training sırasında normalized obs görüyor, eval sırasında ise neredeyse ham obs görüyor.  
- `EvalCallback` yanlış normalizasyon üzerinden "en iyi model" seçiyordu.  
- Deterministic best_model eval'i: gap-only → 3533 dk, hibrit → 3308 dk (her ikisi de FCFS'ten kötü).  
- Raporlanan "iyi" sonuçlar (3029.6, 2994.8 dk) `TrainingLogger`'ın stochastic eğitim episode minimumlarından geliyordu.

**Fix:**
```python
class EvalVecNormSyncCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.eval_vn.obs_rms = copy.deepcopy(self.train_vn.obs_rms)
        return True
```
`model.learn(callback=[TrainingLogger(), eval_norm_sync, eval_callback], ...)` şeklinde callback listesine eklendi; `eval_norm_sync` her zaman `eval_callback`'ten önce çalışır.

**Sonuç:** Önceki tüm RL eğitim sonuçları geçersiz sayılmalı. ADR-016 mimarisi doğru; fix sonrası temiz eğitim gerekiyor.

---

## ADR-018: benchmark.py İyileştirmeleri

**Karar:**  
1. GA ve ACO importları lazy hale getirildi (`deap` yüklü değilse `--skip ga aco` ile çalışsın)  
2. `--model_dir` parametresi eklendi (default: `models`)  
3. `run_rl()` içinde best_model için `vec_normalize_best.pkl` öncelikli olarak aranıyor

**Neden:**  
- Önceki kodda GA/ACO top-level import'tan geliyordu; `deap` kurulu olmayan ortamda tüm benchmark hata veriyordu.  
- Farklı `save_dir`'larla eğitilen modelleri karşılaştırmak için model dizini CLI'dan belirtilmeli.  
- best_model ile final model'in farklı VecNormalize istatistiklerine sahip olduğu ortaya çıktı — doğru `.pkl`'nin otomatik seçilmesi gerekiyordu.

---

## ADR-019: Hiperparametre Tuning — LR Schedule, ent_coef, n_steps

**Karar:** Aşamalı tuning sonucu belirlenen optimal hiperparametreler:
```python
learning_rate = LinearSchedule(3e-4, 5e-5, 1.0)  # sabit 3e-4'ten schedule'a geçiş
ent_coef = 0.1        # 0.05'ten artırıldı
n_steps  = 4096       # 2048'den artırıldı
```

**Neden — Aşamalı deneme süreci:**

1. **LR schedule (3e-4→5e-5):** Sabit LR ile model step 120k'da iyi bir noktayı bulup collapse ediyordu. Schedule ile collapse step 340k'ya ertelendi, best eval -429.03'e iyileşti.

2. **ent_coef=0.1:** Model iyi sekansları stochastic keşif sırasında bulabiliyor ama deterministic politikaya yansıtamıyordu. Entropi artırılınca keşif uzadı, politika daha iyi oturdu. Best eval -417.69'a iyileşti; collapse ortadan kalktı, 500k boyunca iyileşme devam etti.

3. **n_steps=4096:** En büyük sıçramayı sağladı. Her güncellemede daha fazla episode verisi (~10 episode) toplanınca credit assignment güçlendi. Best eval **-364.17**, deterministic gecikme **3024.9 dk** — FCFS (3040.8 dk) geçildi.

**Sonuç:** 500k adımda FCFS geçildi. Training episode minimumları 3020-3027 dk bandına indi. 2M adımda TS/SA seviyesi (~3001 dk) hedefleniyor.