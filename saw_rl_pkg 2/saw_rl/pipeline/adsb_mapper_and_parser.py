"""
SAW ADS-B Mapper & Parser (v2 — tshark direct)
================================================
pyshark yerine tshark'ı doğrudan subprocess olarak çağırır.
Filtreleme C'de yapıldığından 5-10x daha hızlıdır.

Gereksinim: tshark sistem PATH'inde olmalı.
  macOS:  brew install wireshark
  Ubuntu: sudo apt install tshark

Kullanım:
    python -m saw_rl.pipeline.adsb_mapper_and_parser kayit.pcapng
"""

import datetime
import json
import os
import random
import subprocess
import sys
from tqdm import tqdm


def generate_kml_color() -> str:
    r = lambda: random.randint(0, 255)
    return f"bf{r():02x}{r():02x}{r():02x}"


def _check_tshark() -> str:
    """tshark'ın PATH'de olduğunu doğrula, tam yolu döndür."""
    import shutil
    path = shutil.which("tshark")
    if not path:
        raise RuntimeError(
            "tshark bulunamadı.\n"
            "  macOS : brew install wireshark\n"
            "  Ubuntu: sudo apt install tshark"
        )
    return path


def adsb_total_exporter(pcap_file: str) -> None:
    if not os.path.exists(pcap_file):
        print(f"Hata: '{pcap_file}' dosyası bulunamadı!")
        return

    tshark = _check_tshark()
    base_name   = os.path.splitext(pcap_file)[0]
    output_txt  = f"{base_name}_rapor.txt"
    output_kml  = f"{base_name}_rota.kml"

    # ── tshark çağrısı ──────────────────────────────────────────
    # -Y : display filter (sadece aircraft içeren HTTP paketleri)
    # -T fields -e : sadece istediğimiz alanları al
    # frame.time_epoch : unix timestamp (float)
    # http.file_data   : HTTP body (hex encoded, kolon ayraçlı)
    cmd = [
        tshark,
        "-r", pcap_file,
        "-Y", 'http.file_data contains "aircraft"',
        "-T", "fields",
        "-e", "frame.time_epoch",
        "-e", "http.file_data",
        "-E", "separator=|",   # alan ayraçı — JSON içindeki virgülle çakışmasın
    ]

    print(f"\nAnaliz Başlatıldı: {pcap_file}")
    print(f"tshark filtresi çalışıyor...")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # tshark uyarılarını gizle
            text=True,
            bufsize=1,
        )
    except Exception as e:
        print(f"tshark başlatılamadı: {e}")
        return

    all_data_points = []
    paths           = {}
    pbar            = tqdm(desc="İşleniyor", unit=" paket")

    for line in proc.stdout:
        pbar.update(1)
        line = line.strip()
        if not line or "|" not in line:
            continue

        parts = line.split("|", 1)
        if len(parts) != 2:
            continue

        raw_ts, hex_data = parts
        try:
            timestamp = float(raw_ts)
        except ValueError:
            continue

        # tshark hex_data: "7b:22:6e:6f:77..." → bytes → str
        try:
            raw_bytes = bytes.fromhex(hex_data.replace(":", ""))
            json_str  = raw_bytes.decode("utf-8", errors="ignore")
        except Exception:
            continue

        # Bazen tshark birden fazla HTTP chunk'ı birleştirir;
        # ilk geçerli JSON objesini al
        try:
            json_data = json.loads(json_str)
        except json.JSONDecodeError:
            # Truncated chunk — ilk { ... } bloğunu bulmayı dene
            start = json_str.find("{")
            end   = json_str.rfind("}") + 1
            if start == -1 or end == 0:
                continue
            try:
                json_data = json.loads(json_str[start:end])
            except Exception:
                continue

        if "aircraft" not in json_data:
            continue

        abs_time = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")

        for ac in json_data["aircraft"]:
            icao = ac.get("hex")
            if not icao:
                continue

            ac["abs_time"] = abs_time
            all_data_points.append(ac)

            lat, lon = ac.get("lat"), ac.get("lon")
            if lat and lon:
                alt_m = ac.get("alt_baro", 0) * 0.3048
                paths.setdefault(icao, []).append((lon, lat, alt_m))

    pbar.close()
    proc.wait()

    if not all_data_points:
        print("\nUyarı: Dosyada işlenecek ADS-B verisi bulunamadı.")
        return

    print(f"\nToplam kayıt: {len(all_data_points):,} | Uçak: {len(paths):,}")
    print("Dosyalar yazılıyor...")

    # ── TXT çıktısı ─────────────────────────────────────────────
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"ADS-B EKSİKSİZ VERİ DÖKÜMÜ - Kaynak: {pcap_file}\n")
        f.write("=" * 80 + "\n\n")
        for i, entry in enumerate(all_data_points):
            f.write(f"KAYIT #{i+1} | ZAMAN: {entry.get('abs_time')}\n")
            f.write("-" * 40 + "\n")
            for key, value in sorted(entry.items()):
                if key != "abs_time":
                    f.write(f"{key:<15}: {value}\n")
            f.write("\n" + "*" * 50 + "\n\n")

    # ── KML çıktısı ─────────────────────────────────────────────
    with open(output_kml, "w", encoding="utf-8") as k:
        k.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        k.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n')
        k.write(f"  <name>Rotalar - {base_name}</name>\n")
        for icao, points in paths.items():
            color = generate_kml_color()
            k.write(f"  <Placemark>\n    <name>{icao.upper()}</name>\n")
            k.write(f"    <Style><LineStyle><color>{color}</color>"
                    f"<width>4</width></LineStyle></Style>\n")
            k.write("    <LineString>\n")
            k.write("      <extrude>1</extrude>\n")
            k.write("      <altitudeMode>absolute</altitudeMode>\n")
            coords = " ".join(f"{p[0]},{p[1]},{p[2]}" for p in points)
            k.write(f"      <coordinates>{coords}</coordinates>\n")
            k.write("    </LineString>\n  </Placemark>\n")
        k.write("</Document>\n</kml>\n")

    print(f"Bitti!\n  → {output_txt}\n  → {output_kml}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python -m saw_rl.pipeline.adsb_mapper_and_parser dosya.pcapng")
    else:
        adsb_total_exporter(sys.argv[1])
