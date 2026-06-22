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
                    # Header
                    writer.writerow([
                        "Waktu_Terima", "Tipe", "EPC_RFID",
                        "RSSI_1", "RSSI_2", "Plat_OCR",
                        "Gol_Kamera", "Timestamp_Verif",
                        "Delay_Antar_Tag", "Delay_E2E"
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

                if not raw_line or raw_line.startswith(">> [INFO]"):
                    if raw_line:
                        print(f"\033[93m{raw_line}\033[0m")
                    continue

                parts = raw_line.split('|')
                tipe  = parts[0]

                waktu_terima     = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                tanggal_hari_ini = datetime.now().strftime('%d/%m/%Y')

                if tipe == "DATA" and len(parts) >= 6:
                    # Format lama dari firmware ESP — pecah campuran
                    epc_rfid        = parts[1]
                    rssi_jarak1     = parts[2]
                    rssi_jarak2     = parts[3]
                    plat_ocr        = parts[4]
                    campuran        = parts[5]

                    sub             = campuran.split(',')
                    golongan_kamera = sub[0] if len(sub) >= 1 else "N/A"
                    timestamp_jam   = sub[1] if len(sub) >= 2 else "N/A"
                    t1_str          = sub[2] if len(sub) >= 3 else "N/A"
                    delay_antar_tag = sub[3] if len(sub) >= 4 else "N/A"

                    try:
                        delay_e2e = round(time.time() - float(t1_str), 3)
                    except:
                        delay_e2e = "N/A"

                    timestamp_verif = f"{timestamp_jam} {tanggal_hari_ini}"

                    if tipe_data == "RFID":
                        print("\n[JALUR UTAMA - EPC]")
                        # query DB by EPC → payment

                    elif tipe_data == "CAM":
                        print("\n[JALUR CADANGAN - KAMERA]")
                        # query DB by plat → payment

                    elif tipe_data == "FULL":
                        print("\n[AUDIT - GABUNGAN]")
                        # validasi silang + log

                    print(f" TIPE           : {tipe_data}")
                    print(f" WAKTU TERIMA   : {waktu_terima}")
                    print(f" EPC RFID       : {epc_rfid}")
                    print(f" RSSI Jarak 1   : {rssi_jarak1} dBm")
                    print(f" RSSI Jarak 2   : {rssi_jarak2} dBm")
                    print(f" PLAT OCR       : {plat_ocr}")
                    print(f" GOL KAMERA     : {golongan_kamera}")
                    print(f" DELAY ANTAR TAG: {delay_antar_tag}s")
                    print(f" DELAY E2E      : {delay_e2e}s")
                    print(f" TIMESTAMP      : {timestamp_verif}")
                    print("=" * 50)

                    if SAVE_TO_CSV:
                        with open(CSV_FILENAME, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                waktu_terima, tipe_data, epc_rfid,
                                rssi_jarak1, rssi_jarak2, plat_ocr,
                                golongan_kamera, timestamp_verif,
                                delay_antar_tag, delay_e2e
                            ])
                        print(f" [+] Tersimpan ke {CSV_FILENAME}!")

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