import pandas as pd
import numpy as np
import os
import sys
from datetime import timedelta

SAW_LAT = 40.8986
SAW_LON = 29.3092

# ---------------------------
# VEKTÖRLEŞTİRİLMİŞ GEO UTILS
# ---------------------------
def haversine_vec(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def calculate_bearing_vec(lat1, lon1, lat2, lon2):
    l1, l2 = np.radians(lat1), np.radians(lat2)
    diff_lon = np.radians(lon2 - lon1)
    x = np.sin(diff_lon) * np.cos(l2)
    y = np.cos(l1)*np.sin(l2) - np.sin(l1)*np.cos(l2)*np.cos(diff_lon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

# ---------------------------
# HIZLANDIRILMIŞ PIPELINE (KURŞUN GEÇİRMEZ V3)
# ---------------------------
def adsb_saw_sequencer_v3_fast(file_path):
    if not os.path.exists(file_path):
        print(f"Hata: '{file_path}' dosyası bulunamadı!")
        return

    print("[*] Veriler okunuyor...")
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        current = {}
        for line in f:
            line = line.strip()
            if "KAYIT #" in line and "ZAMAN:" in line:
                if current: records.append(current)
                current = {'abs_time': line.split("ZAMAN: ")[1]}
            elif ":" in line and not any(x in line for x in ["===", "***", "---"]):
                k, v = line.split(":", 1)
                current[k.strip()] = v.strip()
        if current: records.append(current)

    df = pd.DataFrame(records)
    df['abs_time'] = pd.to_datetime(df['abs_time'])
    
    num_cols = ['alt_baro','baro_rate','gs','lat','lon','track', 'rssi', 'nac_p']
    for c in num_cols:
        df[c] = pd.to_numeric(df.get(c), errors='coerce')

    df = df.sort_values('abs_time')

    print("[*] Analiz yapılıyor (Vektörel)...")
    summary = df.groupby('hex').agg(
        last_seen=('abs_time','last'),
        alt_first=('alt_baro', 'first'),
        alt_last=('alt_baro','last'),
        baro_rate_med=('baro_rate','median'),
        gs_avg=('gs', lambda x: x.dropna().clip(120, 600).tail(10).median() if not x.dropna().empty else 400),
        lat_last=('lat','last'),
        lon_last=('lon','last'),
        track_last=('track','last'),
        flight=('flight','last'),
        category=('category', 'last'),     
        rssi_last=('rssi', 'last'),        
        nac_p_last=('nac_p', 'last')       
    ).reset_index()

    # Coğrafi Hesaplamalar
    summary['dist_km'] = haversine_vec(summary['lat_last'], summary['lon_last'], SAW_LAT, SAW_LON)
    summary['bearing'] = calculate_bearing_vec(summary['lat_last'], summary['lon_last'], SAW_LAT, SAW_LON)
    
    # Heading Diff
    h_diff = np.abs(summary['track_last'] - summary['bearing'])
    summary['hdg_diff'] = np.minimum(h_diff, 360 - h_diff).fillna(180)

    # 1. FAZ TESPİTİ
    alt_diff = summary['alt_last'] - summary['alt_first'].fillna(summary['alt_last'])
    rate = summary['baro_rate_med'].fillna(0)
    dist = summary['dist_km']
    alt = summary['alt_last'].fillna(30000)

    # Faz seçimleri
    def determine_tma(row_alt_diff, row_rate, row_alt):
        if row_alt_diff < -1000 or row_rate < -150: return "ARRIVAL"
        if row_alt_diff > 1000 or row_rate > 150: return "DEPARTURE"
        return "ARRIVAL" if row_alt < 10000 else "DEPARTURE"

    # Fazları vektörel belirle
    summary['phase'] = "CRUISE"
    
    # Piste yakınlık (Final)
    mask_final = (dist < 25) & (alt < 8000)
    summary.loc[mask_final, 'phase'] = np.where((alt_diff[mask_final] < 0) | (rate[mask_final] < -50), "ARRIVAL", 
                                       np.where((alt_diff[mask_final] > 0) | (rate[mask_final] > 50), "DEPARTURE", "ARRIVAL"))
    
    # TMA Bölgesi
    mask_tma = (dist < 80) & (alt < 18000) & (summary['phase'] == "CRUISE")
    summary.loc[mask_tma, 'phase'] = np.vectorize(determine_tma)(alt_diff[mask_tma], rate[mask_tma], alt[mask_tma])

    # Diğerleri
    mask_cruise = (alt > 24000) & (np.abs(rate) < 200) & (np.abs(alt_diff) < 2000)
    summary.loc[mask_cruise & (summary['phase'] == "CRUISE"), 'phase'] = "CRUISE"
    
    summary.loc[(summary['phase'] == "CRUISE") & ((rate < -200) | (alt_diff < -2000)), 'phase'] = "ARRIVAL"
    summary.loc[(summary['phase'] == "CRUISE") & ((rate > 200) | (alt_diff > 2000)), 'phase'] = "DEPARTURE"

    # 2. ETA HESABI (YAMALANMIŞ BÖLÜM)
    summary['target_time'] = summary['last_seen']
    
    # Mikro Offset
    micro_offset = summary['hex'].apply(lambda x: int(str(int(x, 16))[-4:]) % 1000).astype(int)
    
    gs_km_min = (summary['gs_avg'] * 1.852) / 60
    t_h = summary['dist_km'] / np.maximum(gs_km_min, 2)
    t_v = alt / np.maximum(np.abs(rate), 500)
    
    # Varışlar için ETA - BÜYÜK DÜZELTME BURADA: np.fmax kullanıyoruz!
    arr_mask = summary['phase'].isin(['ARRIVAL', 'FINAL'])
    eta_min_arr = np.fmax(t_h[arr_mask], t_v[arr_mask])
    
    summary.loc[arr_mask, 'target_time'] += pd.to_timedelta(eta_min_arr, unit='m')
    
    # Kalkışlar için Geriye Dönük Hesap
    dep_mask = summary['phase'] == 'DEPARTURE'
    summary.loc[dep_mask, 'target_time'] -= pd.to_timedelta(t_v[dep_mask], unit='m') + pd.to_timedelta(20, unit='s')
    
    # Tümüne mikro offset ekle
    summary['target_time'] += pd.to_timedelta(micro_offset, unit='ms')

    # State Encoding ve Çıktı
    def encode_cat_vec(c):
        c = str(c).upper()
        for i in ['5','4','3','2','1']:
            if 'A'+i in c: return int(i)
        return 0

    # RL Ajanı İçin NaN Korumalı State Matrisi
    summary['state'] = summary.apply(lambda r: [
        round(r['dist_km'], 2) if pd.notna(r['dist_km']) else 50.0, # NaN ise varsayılan 50km
        round(r['alt_last'], 0) if pd.notna(r['alt_last']) else 10000.0,
        round(r['gs_avg'], 1) if pd.notna(r['gs_avg']) else 250.0,
        round(r['hdg_diff'], 1) if pd.notna(r['hdg_diff']) else 180.0, 
        r['baro_rate_med'] if pd.notna(r['baro_rate_med']) else 0.0, 
        encode_cat_vec(r['category']), 
        r['rssi_last'] if pd.notna(r['rssi_last']) else -20.0
    ], axis=1)

    # 3. KARA DELİK FİLTRESİ: Ne yatay ne dikey verisi olan çöp satırları at!
    summary = summary.dropna(subset=['target_time']).copy()

    base = os.path.splitext(file_path)[0]
    summary[summary['phase'].isin(['ARRIVAL', 'FINAL'])].to_csv(base+"_saw_arrivals.csv", index=False)
    summary[summary['phase'] == 'DEPARTURE'].to_csv(base+"_saw_departures.csv", index=False)
    
    print(f"\n✅ İŞLEM TAMAMLANDI. Toplam Geçerli Uçak: {len(summary)}")

if __name__ == "__main__":
    file = sys.argv[1] if len(sys.argv) > 1 else "adsb1_rapor.txt"
    adsb_saw_sequencer_v3_fast(file)