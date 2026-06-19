#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
from datetime import datetime

# === KONFIGURASI ===
TIME_TOLERANCE = 0.5  # toleransi waktu (detik) untuk mencocokkan data RFID dan kamera
MAX_DETEKSI    = 3
COOLDOWN_LOCK  = 10
TIMEOUT_KAMERA = 1.0   # waktu maksimum menunggu data kamera sebelum dianggap timeout

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

        # ↓ Publisher ke sensor.py (bukan serial langsung)
        self.pub_kirim = self.create_publisher(String, '/kirim_ke_esp', 10)

        self.create_subscription(String, '/rfid_data',   self.callback_rfid,   10)
        self.create_subscription(String, '/kamera_data', self.callback_kamera, 10)

        self.create_timer(0.2, self.cek_timeout_kamera) #waktu cek timeout kamera setiap 0.2 detik

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

    def tampilkan_rfid(self, epc, rssi, timestamp_rfid="N/A", delay="N/A"):
        sep = "-" * 55
        self.get_logger().info(sep)
        self.get_logger().info(f"[RFID] EPC       : {epc}")
        self.get_logger().info(f"[RFID] RSSI      : {rssi}")
        self.get_logger().info(f"[RFID] Timestamp : {timestamp_rfid}")
        self.get_logger().info(f"[RFID] Delay     : {delay}s")
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
    

    def kirim_ke_esp(self, epc, rssi, plat_ocr, plat_db, golongan_kamera, golongan_db, status, timestamp, t1="N/A", delay_antar_tag="N/A"):
        rssi_angka = str(rssi).replace(" dBm", "").strip()

        status_map = {
            "VALID": "VAL", "INVALID": "INV",
            "UNKNOWN": "UNK", "NO_CAMERA": "NOCAM"
        }
        status_kode       = status_map.get(status, "UNK")
        plat_db_kode      = "-" if plat_db == "TIDAK DITEMUKAN" else plat_db
        timestamp_singkat = timestamp.split(" ")[0]

        campuran  = f"{plat_db_kode},{golongan_db},{status_kode},{timestamp_singkat},{t1},{delay_antar_tag}"
        pesan_esp = f"{epc}|{rssi_angka}|{plat_ocr}|{campuran}"

        msg = String()
        msg.data = pesan_esp
        self.pub_kirim.publish(msg)
        self.get_logger().info(f"[KIRIM ESP] {pesan_esp}")

    # def kirim_ke_esp(self, epc, rssi, plat_ocr, plat_db, golongan_kamera, golongan_db, status, timestamp, delay="N/A"):
    #     rssi_angka = str(rssi).replace(" dBm", "").strip()

    #     status_map = {
    #         "VALID": "VAL",
    #         "INVALID": "INV",
    #         "UNKNOWN": "UNK",
    #         "NO_CAMERA": "NOCAM"
    #     }
    #     status_kode = status_map.get(status, "UNK")

    #     plat_db_kode = "-" if plat_db == "TIDAK DITEMUKAN" else plat_db
    #     timestamp_singkat = timestamp.split(" ")[0]

    #     # ↓ TAMBAH delay ke campuran
    #     campuran  = f"{plat_db_kode},{golongan_db},{status_kode},{timestamp_singkat},{delay}"
    #     pesan_esp = f"{epc}|{rssi_angka}|{plat_ocr}|{campuran}"

    #     msg = String()
    #     msg.data = pesan_esp
    #     self.pub_kirim.publish(msg)
    #     self.get_logger().info(f"[KIRIM ESP] {pesan_esp}")

    def verifikasi(self, rfid_data, kamera_data):
        epc             = rfid_data["epc"]
        rssi            = rfid_data.get("rssi", "N/A")
        #delay           = rfid_data.get("delay", "N/A")   # ← TAMBAH INI
        plat_ocr        = kamera_data["plat"]
        golongan_kamera = kamera_data["golongan"]
        timestamp       = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        t1              = rfid_data.get("t1", "N/A")
        delay_antar_tag = rfid_data.get("delay_antar_tag", "N/A")

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

        self.kirim_ke_esp(epc, rssi, plat_ocr, plat_db, golongan_kamera, golongan_db, status, timestamp, t1, delay_antar_tag)
        #self.kirim_ke_esp(epc, rssi, plat_ocr, plat_db, golongan_kamera, golongan_db, status, timestamp, delay)

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

            # ↓ KIRIM TETAP MESKIPUN TIDAK ADA KAMERA
            epc       = r["epc"]
            rssi      = r.get("rssi", "N/A")
            timestamp = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

            db_data = dummy_db.get(epc)
            if db_data:
                plat_db     = db_data["plat"]
                golongan_db = db_data["golongan"]
            else:
                plat_db     = "TIDAK DITEMUKAN"
                golongan_db = "-"

            status = "NO_CAMERA"  # status khusus: RFID terbaca tanpa verifikasi kamera

            self.kirim_ke_esp(
                epc, rssi, "N/A", plat_db, "-", golongan_db, status, timestamp,
                r.get("t1", "N/A"),
                r.get("delay_antar_tag", "N/A")
            )

            # self.kirim_ke_esp(
            #     epc, rssi,
            #     "N/A", plat_db, "-", golongan_db, status, timestamp,
            #     r.get("delay", "N/A")   # ← TAMBAH INI
            # )

    def callback_rfid(self, msg):
        raw   = msg.data.strip()
        parts = raw.split(",")
        epc             = parts[0]
        rssi            = parts[1] if len(parts) >= 2 else "N/A"
        t1              = parts[2] if len(parts) >= 3 else "N/A"
        delay_antar_tag = parts[3] if len(parts) >= 4 else "N/A"
        # raw   = msg.data.strip()
        # parts = raw.split(",")
        # epc            = parts[0]
        # rssi           = parts[1] if len(parts) >= 2 else "N/A"
        # timestamp_rfid = parts[2] if len(parts) >= 3 else "N/A"
        # delay          = parts[3] if len(parts) >= 4 else "N/A"

        if not self.cek_limit(epc):
            return

        self.tampilkan_rfid(epc, rssi)
        #self.tampilkan_rfid(epc, rssi, timestamp_rfid, delay)

        self.rfid_buffer.append({
            "epc":            epc,
            "rssi":           rssi,
            #"timestamp_rfid": timestamp_rfid,
            "t1":              t1,
            "delay_antar_tag": delay_antar_tag,
            # "delay":          delay,           # ← pastikan ada ini
            "timestamp":      time.time()
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