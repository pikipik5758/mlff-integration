#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import serial                          # ← TAMBAH INI
from datetime import datetime

# === KONFIGURASI ===
TIME_TOLERANCE = 2.0
MAX_DETEKSI    = 3
COOLDOWN_LOCK  = 10
TIMEOUT_KAMERA = 2.0

# === DUMMY DATABASE ===
dummy_db = {
    "E28069150000700B41525446": {"plat": "L 1829 ABO", "golongan": 1},
    "E28069150000600B41554045": {"plat": "W 1940 VI",  "golongan": 4},
    "E28069150000600B41524446": {"plat": "B 448 ALI",  "golongan": 1},
}


def similarity(a, b):
    a = a.replace(" ", "").upper()
    b = b.replace(" ", "").upper()
    if not a or not b:
        return 0.0
    match = sum(c1 == c2 for c1, c2 in zip(a, b))
    return match / max(len(a), len(b))


class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')

        self.rfid_buffer   = []
        self.kamera_buffer = []
        self.status_tag    = {}

        # Serial kirim ke ESP32-S3
        self.ser_kirim = None
        try:
            self.ser_kirim = serial.Serial()
            self.ser_kirim.port     = '/dev/ttyACM0'
            self.ser_kirim.baudrate = 115200
            self.ser_kirim.timeout  = 1
            self.ser_kirim.dtr      = False
            self.ser_kirim.rts      = False
            self.ser_kirim.open()
            time.sleep(0.5)
            self.get_logger().info("Serial kirim ke ESP32-S3 siap")
        except Exception as e:
            self.get_logger().warn(f"Serial kirim tidak terhubung: {e}")
            self.ser_kirim = None

        self.create_subscription(String, '/rfid_data',   self.callback_rfid,   10)
        self.create_subscription(String, '/kamera_data', self.callback_kamera, 10)

        self.create_timer(1.0, self.cek_timeout_kamera)

        self.get_logger().info("Subscriber node siap — menunggu data RFID & Kamera...")

    def cek_limit(self, epc):
        waktu_sekarang = time.time()
        if epc not in self.status_tag:
            self.status_tag[epc] = [1, waktu_sekarang]
            return True
        jumlah, waktu_lama = self.status_tag[epc]
        if jumlah >= MAX_DETEKSI and (waktu_sekarang - waktu_lama < COOLDOWN_LOCK):
            self.get_logger().warn(f"[SKIP] Tag {epc} cooling down...")
            return False
        if waktu_sekarang - waktu_lama >= COOLDOWN_LOCK:
            self.status_tag[epc] = [1, waktu_sekarang]
        else:
            self.status_tag[epc][0] += 1
            self.status_tag[epc][1]  = waktu_sekarang
        return True

    def tampilkan_rfid(self, epc, rssi):
        sep = "-" * 55
        self.get_logger().info(sep)
        self.get_logger().info(f"[RFID] EPC  : {epc}")
        self.get_logger().info(f"[RFID] RSSI : {rssi}")
        self.get_logger().info(f"[RFID] Menunggu data kamera untuk verifikasi...")
        self.get_logger().info(sep)

    def coba_cocokkan(self):
        matched_rfid   = None
        matched_kamera = None

        for r in self.rfid_buffer:
            for k in self.kamera_buffer:
                if abs(r["timestamp"] - k["timestamp_ros"]) <= TIME_TOLERANCE:
                    matched_rfid   = r
                    matched_kamera = k
                    break
            if matched_rfid:
                break

        if not matched_rfid or not matched_kamera:
            return

        self.rfid_buffer.remove(matched_rfid)
        self.kamera_buffer.remove(matched_kamera)
        self.verifikasi(matched_rfid, matched_kamera)

    def verifikasi(self, rfid_data, kamera_data):
        epc             = rfid_data["epc"]
        rssi            = rfid_data.get("rssi", "N/A")
        plat_ocr        = kamera_data["plat"]
        golongan_kamera = kamera_data["golongan"]
        timestamp       = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

        db_data = dummy_db.get(epc)
        if db_data:
            plat_db     = db_data["plat"]
            golongan_db = db_data["golongan"]
            sim         = similarity(plat_ocr, plat_db)
            plat_valid  = sim >= 0.6
            gol_valid   = golongan_kamera == golongan_db
            status      = "VALID" if plat_valid and gol_valid else "INVALID"
        else:
            plat_db     = "TIDAK DITEMUKAN"
            golongan_db = "-"
            status      = "UNKNOWN"

        sep = "=" * 55
        self.get_logger().info(sep)
        self.get_logger().info(f"[VERIFIKASI] EPC         : {epc}")
        self.get_logger().info(f"[VERIFIKASI] RSSI        : {rssi}")
        self.get_logger().info(f"[VERIFIKASI] Plat OCR    : {plat_ocr}")
        self.get_logger().info(f"[VERIFIKASI] Plat DB     : {plat_db}")
        self.get_logger().info(f"[VERIFIKASI] Gol Kamera  : {golongan_kamera}")
        self.get_logger().info(f"[VERIFIKASI] Gol DB      : {golongan_db}")
        self.get_logger().info(f"[VERIFIKASI] Status      : {status}")
        self.get_logger().info(f"[VERIFIKASI] Waktu       : {timestamp}")
        self.get_logger().info(sep)

        # ↓ KIRIM KE ESP32-S3 → ESP-NOW → ESP32 TEMAN
        if self.ser_kirim and self.ser_kirim.is_open:
            rssi_angka = rssi.replace(" dBm", "").strip()
            campuran   = f"{plat_db},{golongan_db},{status},{timestamp}"
            pesan_esp  = f"{epc}|{rssi_angka}|{plat_ocr}|{campuran}\n"
            self.ser_kirim.write(pesan_esp.encode('utf-8'))
            self.get_logger().info(f"[KIRIM ESP] {pesan_esp.strip()}")

    def cek_timeout_kamera(self):
        waktu_sekarang = time.time()
        expired = [
            r for r in self.rfid_buffer
            if waktu_sekarang - r["timestamp"] > TIMEOUT_KAMERA
        ]
        for r in expired:
            self.rfid_buffer.remove(r)
            self.get_logger().warn(
                f"[TIMEOUT] EPC {r['epc']} — "
                f"kamera tidak datang dalam {TIMEOUT_KAMERA}s, dibatalkan."
            )

    def callback_rfid(self, msg):
        raw   = msg.data.strip()
        parts = raw.split(",")
        epc   = parts[0]
        rssi  = parts[1] if len(parts) >= 2 else "N/A"

        if not self.cek_limit(epc):
            return

        self.tampilkan_rfid(epc, rssi)

        self.rfid_buffer.append({
            "epc":       epc,
            "rssi":      rssi,
            "timestamp": time.time()
        })

        self.coba_cocokkan()

    def callback_kamera(self, msg):
        try:
            data = json.loads(msg.data)
            data["timestamp_ros"] = time.time()

            waktu_sekarang = time.time()
            self.kamera_buffer = [
                k for k in self.kamera_buffer
                if waktu_sekarang - k["timestamp_ros"] <= TIME_TOLERANCE
            ]

            self.kamera_buffer.append(data)
            self.get_logger().info(f"[KAMERA] Plat: {data['plat']} | Gol: {data['golongan']}")
            self.coba_cocokkan()
        except Exception as e:
            self.get_logger().warn(f"Error parse kamera: {e}")

    # ↓ TAMBAH FUNGSI INI
    def destroy_node(self):
        if self.ser_kirim and self.ser_kirim.is_open:
            self.ser_kirim.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()