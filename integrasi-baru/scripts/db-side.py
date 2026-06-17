import serial
import time
import mysql.connector
from datetime import datetime
import csv
import os

# ==========================================
# 1. PENGATURAN KONEKSI SERIAL
# ==========================================
COM_PORT = 'COM9'
BAUD_RATE = 115200

# ==========================================
# 2. PENGATURAN KONEKSI DATABASE XAMPP
# ==========================================
USE_DATABASE = False

if USE_DATABASE:
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="db_tol_mlff"
        )
        cursor = db.cursor()
        print("[INFO] Berhasil terhubung ke Database MySQL XAMPP.")
    except Exception as e:
        print(f"[ERROR DB] Gagal terhubung ke database: {e}")
        USE_DATABASE = False

# ==========================================
# 3. PENGATURAN SIMPAN OTOMATIS (CSV)
# ==========================================
SAVE_TO_CSV = True
CSV_FILENAME = "rekaman_pengujian_mlff_6_delay.csv"


# ==========================================
# 4. PROGRAM UTAMA
# ==========================================
def main():
    global SAVE_TO_CSV

    if SAVE_TO_CSV:
        file_exists = os.path.isfile(CSV_FILENAME)
        try:
            with open(CSV_FILENAME, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow([
                        "Waktu_Terima", "EPC_RFID", "RSSI_1", "RSSI_2", "Plat_OCR",
                        "Plat_DB", "Golongan_DB", "Status", "Timestamp_Verifikasi", "Delay_RFID"
                    ])
            print(f"[INFO] Fitur rekam otomatis aktif. Data akan disimpan di: {CSV_FILENAME}")
        except Exception as e:
            print(f"[ERROR] Gagal membuat file CSV: {e}")
            SAVE_TO_CSV = False

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Mendengarkan Port {COM_PORT} pada Baud {BAUD_RATE}...")
        print("[INFO] Menunggu data integrasi MLFF dari Gerbang...\n")

        while True:
            if ser.in_waiting > 0:
                raw_line = ser.readline().decode('utf-8', errors='ignore').strip()

                if not raw_line:
                    continue

                if raw_line.startswith(">>> [ANALISA]"):
                    print(f"\033[93m{raw_line}\033[0m")
                    continue

                parts = raw_line.split('|')

                if parts[0] == "DATA" and len(parts) >= 6:
                    epc_rfid    = parts[1]
                    rssi_jarak1 = parts[2]
                    rssi_jarak2 = parts[3]
                    plat_ocr    = parts[4]
                    campuran    = parts[5]

                    sub_parts = campuran.split(',')
                    plat_db_kode  = sub_parts[0] if len(sub_parts) >= 1 else "N/A"
                    golongan_db   = sub_parts[1] if len(sub_parts) >= 2 else "N/A"
                    status_kode   = sub_parts[2] if len(sub_parts) >= 3 else "UNK"
                    timestamp_jam = sub_parts[3] if len(sub_parts) >= 4 else "N/A"
                    delay         = sub_parts[4] if len(sub_parts) >= 5 else "N/A"

                    status_map_balik = {
                        "VAL":   "VALID",
                        "INV":   "INVALID",
                        "UNK":   "UNKNOWN",
                        "NOCAM": "NO_CAMERA"
                    }
                    plat_db = "TIDAK DITEMUKAN" if plat_db_kode == "-" else plat_db_kode
                    status  = status_map_balik.get(status_kode, "UNKNOWN")

                    waktu_terima      = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    tanggal_hari_ini  = datetime.now().strftime('%d/%m/%Y')
                    timestamp_verif   = f"{timestamp_jam} {tanggal_hari_ini}"

                    # Cetak ke Terminal
                    print("\n" + "=" * 50)
                    print(f" WAKTU TERIMA      : {waktu_terima}")
                    print(f" EPC RFID          : {epc_rfid}")
                    print(f" RSSI Jarak 1      : {rssi_jarak1} dBm")
                    print(f" RSSI Jarak 2      : {rssi_jarak2} dBm")
                    print(f" PLAT OCR          : {plat_ocr}")
                    print(f" PLAT DB           : {plat_db}")
                    print(f" GOLONGAN DB       : {golongan_db}")
                    print(f" STATUS            : {status}")
                    print(f" TIMESTAMP VERIF   : {timestamp_verif}")
                    print(f" DELAY (RFID)      : {delay}s")
                    print("=" * 50)

                    # A. Simpan ke DB (jika aktif)  ← TETAP DI SINI, di dalam while/if
                    if USE_DATABASE:
                        sql = """INSERT INTO log_transaksi
                                 (waktu, epc, rssi_1, rssi_2, plat_ocr, plat_db, golongan_db, status, timestamp_verifikasi, delay)
                                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                        val = (waktu_terima, epc_rfid, rssi_jarak1, rssi_jarak2, plat_ocr, plat_db, golongan_db, status, timestamp_verif, delay)
                        try:
                            cursor.execute(sql, val)
                            db.commit()
                            print(" [+] Tersimpan ke Database XAMPP!")
                        except Exception as e:
                            print(f" [-] Gagal menyimpan ke DB: {e}")

                    # B. Simpan ke CSV (jika aktif)
                    if SAVE_TO_CSV:
                        try:
                            with open(CSV_FILENAME, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([
                                    waktu_terima, epc_rfid, rssi_jarak1, rssi_jarak2, plat_ocr,
                                    plat_db, golongan_db, status, timestamp_verif, delay
                                ])
                            print(f" [+] Tersimpan ke {CSV_FILENAME}!")
                        except Exception as e:
                            print(f" [-] Gagal menulis ke CSV: {e}")

                elif raw_line.startswith(">> [INFO]"):
                    print(f"\033[93m{raw_line}\033[0m")

                else:
                    print(f"[RAW] {raw_line}")

    except serial.SerialException:
        print(f"\n[ERROR] Port {COM_PORT} tidak ditemukan atau sedang dipakai program lain.")
    except KeyboardInterrupt:
        print("\n[INFO] Program dihentikan secara manual.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
        if USE_DATABASE and db.is_connected():
            cursor.close()
            db.close()


if __name__ == '__main__':
    main()