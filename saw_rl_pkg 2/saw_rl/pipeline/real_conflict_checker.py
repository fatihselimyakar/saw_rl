import pandas as pd
import sys

class SAWConflictChecker:
    def __init__(self):
        # 🌪️ ICAO Wake Turbulence Ayrım Matrisi
        self.wake_matrix = {
            'HEAVY':  {'HEAVY': 96,  'MEDIUM': 120, 'LIGHT': 180},
            'MEDIUM': {'HEAVY': 60,  'MEDIUM': 90,  'LIGHT': 120},
            'LIGHT':  {'HEAVY': 60,  'MEDIUM': 60,  'LIGHT': 90}
        }
        self.staggered_sep = 45  
        self.rot = 50            
        self.dep_buffer = 15     
        self.min_sep = 10        

    def map_v3_category(self, cat_str):
        cat_str = str(cat_str).upper()
        if 'A5' in cat_str or 'A4' in cat_str: return 'HEAVY'
        elif 'A3' in cat_str: return 'MEDIUM' 
        elif 'A2' in cat_str or 'A1' in cat_str: return 'LIGHT'
        else: return 'MEDIUM' 

    def get_wake_sep(self, lead_cat, follow_cat):
        return self.wake_matrix.get(lead_cat, {}).get(follow_cat, 90)

    def get_separation(self, p_a, p_b):
        if p_a['rwy'] != p_b['rwy']:
            if p_a['type'] != p_b['type']: return self.min_sep, "Farklı Pist: İniş/Kalkış Bağımsız"
            if p_a['type'] == 'ARRIVAL': return max(self.get_wake_sep(p_a['wake_cat'], p_b['wake_cat']), self.staggered_sep), "Farklı Pist: Bağımlı İniş"
            if p_a['type'] == 'DEPARTURE': return self.dep_buffer, "Farklı Pist: Yarı-Bağımsız Kalkış"
        if p_a['type'] == p_b['type']:
            if p_a['type'] == 'ARRIVAL': return self.get_wake_sep(p_a['wake_cat'], p_b['wake_cat']), "Aynı Pist: Peşpeşe İniş"
            if p_a['type'] == 'DEPARTURE': return max(self.get_wake_sep(p_a['wake_cat'], p_b['wake_cat']), 90), "Aynı Pist: Peşpeşe Kalkış"
        if p_a['type'] == 'ARRIVAL' and p_b['type'] == 'DEPARTURE': return self.rot + 20, "Aynı Pist: İniş Sonrası Kalkış"
        if p_a['type'] == 'DEPARTURE' and p_b['type'] == 'ARRIVAL': return 120, "Aynı Pist: Kalkış Sonrası İniş Payı"
        return self.min_sep, "Fallback"

def main():
    if len(sys.argv) < 3:
        print("\nKullanım: python real_atc_checker.py adsb1_saw_arrivals.csv adsb1_saw_departures.csv\n")
        return

    print("[*] Veriler yükleniyor...")
    df_arr = pd.read_csv(sys.argv[1])
    df_dep = pd.read_csv(sys.argv[2])

    # ==========================================
    # 🔥 GERÇEKLEŞEN OPERASYON FİLTRESİ (GROUND TRUTH)
    # ==========================================
    # Sadece fiziksel olarak havalimanı eşiğinde olanları analiz et
    original_arr_len = len(df_arr)
    original_dep_len = len(df_dep)

    # İnenler: Radardaki son irtifası 3000 ft'in altında olanlar (Gerçekten teker koyanlar)
    df_arr = df_arr[df_arr['alt_last'] < 3000].copy()
    
    # Kalkanlar: Radardaki ilk irtifası 3000 ft'in altında olanlar (Gerçekten yeni havalananlar)
    df_dep = df_dep[df_dep['alt_first'] < 3000].copy()

    print(f"[*] Filtreleme: {original_arr_len} İniş talebinden {len(df_arr)} tanesi fiziksel olarak piste inmiş.")
    print(f"[*] Filtreleme: {original_dep_len} Kalkış talebinden {len(df_dep)} tanesi fiziksel olarak pistten kalkmış.")

    checker = SAWConflictChecker()

    df_arr['type'] = 'ARRIVAL'
    df_arr['rwy'] = 1  
    df_dep['type'] = 'DEPARTURE'
    df_dep['rwy'] = 2

    df_arr['wake_cat'] = df_arr['category'].apply(checker.map_v3_category) if 'category' in df_arr.columns else 'MEDIUM'
    df_dep['wake_cat'] = df_dep['category'].apply(checker.map_v3_category) if 'category' in df_dep.columns else 'MEDIUM'

    df_arr['sta'] = pd.to_datetime(df_arr['target_time'])
    df_dep['sta'] = pd.to_datetime(df_dep['target_time'])

    full = pd.concat([df_arr, df_dep]).sort_values('sta').dropna(subset=['sta']).reset_index(drop=True)

    conflicts = []
    total_delay = 0

    print("\n[*] GERÇEK ATC PERFORMANSI TEST EDİLİYOR...\n")
    
    for i in range(len(full)):
        p_a = full.iloc[i]
        for j in range(i+1, len(full)):
            p_b = full.iloc[j]
            dt = (p_b['sta'] - p_a['sta']).total_seconds()

            if dt > 300: break

            req, rule = checker.get_separation(p_a, p_b)

            if dt < req:
                delay = req - dt
                total_delay += delay
                conflicts.append({
                    'Öndeki': f"{p_a['flight']} ({p_a['wake_cat']})",
                    'Arkadaki': f"{p_b['flight']} ({p_b['wake_cat']})",
                    'Mevcut(s)': round(dt, 1),
                    'Gereken(s)': req,
                    'Kural': rule
                })

    print(f"{'='*80}")
    print(f"{'SABİHA GÖKÇEN - GERÇEK ATC OPERASYON RAPORU (GROUND TRUTH)':^80}")
    print(f"{'='*80}")

    if conflicts:
        print(pd.DataFrame(conflicts).to_string(index=False))
        print(f"\n⚠️ Kule (ATC) İhlali: {len(conflicts)}")
        print(f"⏱️ Toplam Gecikme Açığı: {round(total_delay, 1)} saniye")
    else:
        print("\n✅ MÜKEMMEL! Gerçekleşen trafik operasyonlarında hiçbir ayrışma ihlali yoktur.")
        print("Kule (ATC) tüm trafikleri yasal limitler (Wake Turbulence) dahilinde güvenle yönetmiştir.")

    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()