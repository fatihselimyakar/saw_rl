"""
SAW Runway Sequencing — Shared Constants & Utilities
=====================================================
Tüm dosyaların ortak kullandığı sabitler ve yardımcı fonksiyonlar.
Değiştirilecekse SADECE burada değiştir.
"""

# ──────────────────────────────────────────────────────────────
# RECAT-EU Wake Turbulence Separation (saniye)
# Kaynak: EASA RECAT-EU, SAW TWR prosedürleri
# ──────────────────────────────────────────────────────────────
WAKE_SEP: dict[tuple[str, str], int] = {
    ('A5', 'A5'): 120, ('A5', 'A4'): 100, ('A5', 'A3'): 120, ('A5', 'A2'): 140, ('A5', 'A1'): 160,
    ('A4', 'A5'):  80, ('A4', 'A4'): 100, ('A4', 'A3'): 120, ('A4', 'A2'): 120, ('A4', 'A1'): 140,
    ('A3', 'A5'):  80, ('A3', 'A4'):  80, ('A3', 'A3'):  80, ('A3', 'A2'): 100, ('A3', 'A1'): 120,
    ('A2', 'A5'):  80, ('A2', 'A4'):  80, ('A2', 'A3'):  80, ('A2', 'A2'):  80, ('A2', 'A1'): 100,
    ('A1', 'A5'):  80, ('A1', 'A4'):  80, ('A1', 'A3'):  80, ('A1', 'A2'):  80, ('A1', 'A1'):  80,
    ('A0', 'A5'):  80, ('A0', 'A4'):  80, ('A0', 'A3'):  80, ('A0', 'A2'): 100, ('A0', 'A1'): 120,
    ('A5', 'A0'): 120, ('A4', 'A0'): 120, ('A3', 'A0'):  80, ('A2', 'A0'):  80, ('A1', 'A0'):  80,
    ('A0', 'A0'):  80,
}

# Runway Occupancy Time — kategori bazlı (saniye)
# A5/A4: Heavy (B747, A380), A3: Medium-Heavy (B737, A320), A2: Medium, A1: Light
ROT: dict[str, int] = {
    'A5': 65,
    'A4': 65,
    'A3': 55,
    'A2': 45,
    'A1': 38,
    'A0': 55,   # Bilinmeyen → A3 varsayımı
}

# Sayısal encoding (RL state için)
CAT_MAP: dict[str, int] = {
    'A0': 0,
    'A1': 1,
    'A2': 2,
    'A3': 3,
    'A4': 4,
    'A5': 5,
}

# Geçerli kategori kümeleri
VALID_CATS: frozenset[str] = frozenset(CAT_MAP.keys())
DEFAULT_CAT = 'A3'

# SAW Havalimanı koordinatları
SAW_LAT: float = 40.8986
SAW_LON: float = 29.3092

# Bağımlı paralel pisler — staggered separation (saniye)
# 06/24 ile 04/22 arası 1,140m mesafe
STAGGERED_SEP: int = 45

# RL ortamı varsayılan parametreleri
DEFAULT_N_WINDOW: int = 10
DEFAULT_MPS_K: int = 3

# Normalizasyon sınırları [eta_sec, dist, alt, gs, rate, cat, phase, fcfs, hdg]
# Not: fcfs üst sınırı (index 7) runtime'da set edilmeli — sabit bırakıldı
OBS_LOW  = [-600.0,   0.0,     0.0, 100.0, -4000.0, 0.0, 0.0,   0.0,   0.0]
OBS_HIGH = [7200.0, 200.0, 45000.0, 600.0,  4000.0, 5.0, 1.0, 500.0, 180.0]


# ──────────────────────────────────────────────────────────────
# Utility fonksiyonlar
# ──────────────────────────────────────────────────────────────

def get_sep(leader: str, follower: str) -> float:
    """
    RECAT-EU wake turbulence ayrımını ve ROT'u karşılaştırıp
    gerekli minimum ayrımı (saniye) döndürür.

    Args:
        leader:   Öndeki uçağın RECAT kategorisi (örn. 'A3')
        follower: Arkadaki uçağın RECAT kategorisi

    Returns:
        Gerekli minimum zaman ayrımı (saniye, float)
    """
    wake = WAKE_SEP.get((leader, follower), 80)
    rot  = ROT.get(leader, 55)
    return float(max(wake, rot))


def normalize_cat(raw: str) -> str:
    """
    Ham kategori string'ini geçerli bir RECAT koduna normalize eder.
    Bilinmeyen ya da boş değerler DEFAULT_CAT ('A3') olarak döner.

    Args:
        raw: ADS-B verisinden gelen ham kategori string'i

    Returns:
        Normalize edilmiş kategori (örn. 'A3')
    """
    if not raw:
        return DEFAULT_CAT
    raw = str(raw).strip().upper()
    if raw in VALID_CATS:
        return raw
    # "A3MEDIUM" gibi gürültülü veri için prefix kontrolü
    for cat in ('A5', 'A4', 'A3', 'A2', 'A1', 'A0'):
        if cat in raw:
            return cat
    return DEFAULT_CAT


def cat_to_int(cat: str) -> int:
    """Kategori stringini sayısal RL feature'ına dönüştürür."""
    return CAT_MAP.get(normalize_cat(cat), CAT_MAP[DEFAULT_CAT])
