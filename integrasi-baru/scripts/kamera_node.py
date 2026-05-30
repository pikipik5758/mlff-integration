#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR
import numpy as np
import json
import time
import os
import re
import threading
import queue
from datetime import datetime

# === KONFIGURASI ===
MODEL_PATH = "/home/piki/PA/integrasi-baru/scripts/best0895.pt"
CAMERA_INDEX = 2
SAVE_DIR = "/home/piki/PA/integrasi-baru/scripts/hasil"

VEHICLE_CLASSES = {0, 1, 2, 3, 4}
PLATE_CLASS = 5

GOLONGAN_MAP = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
}


# ============================================================
# PREPROCESSING UTILITIES
# ============================================================

def deskew(gray):
    """Koreksi kemiringan plat menggunakan sudut dari minAreaRect."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # Hanya koreksi kalau kemiringan cukup signifikan (> 1 derajat)
    if abs(angle) < 1.0:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def denoise(gray):
    """Hilangkan noise dengan fastNlMeansDenoising."""
    return cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)


def color_segment_plat(crop_bgr):
    """
    Segmentasi warna untuk plat Indonesia:
    - Putih/silver (kendaraan umum & pribadi)
    - Kuning (kendaraan umum/angkutan)
    - Merah (kendaraan dinas pemerintah)
    - Hijau (kendaraan listrik / TNI)
    Mengembalikan mask area teks (gelap) di atas background terang.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    # Deteksi background putih/silver
    mask_putih = cv2.inRange(hsv, (0, 0, 160), (180, 40, 255))
    # Deteksi background kuning
    mask_kuning = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    # Deteksi background merah (dua range karena merah wrap di HSV)
    mask_merah1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    mask_merah2 = cv2.inRange(hsv, (165, 80, 80), (180, 255, 255))
    mask_merah = cv2.bitwise_or(mask_merah1, mask_merah2)
    # Deteksi background hijau
    mask_hijau = cv2.inRange(hsv, (40, 50, 50), (85, 255, 255))

    # Gabungkan semua mask background
    mask_bg = cv2.bitwise_or(mask_putih, mask_kuning)
    mask_bg = cv2.bitwise_or(mask_bg, mask_merah)
    mask_bg = cv2.bitwise_or(mask_bg, mask_hijau)

    # Area teks = bukan background
    mask_teks = cv2.bitwise_not(mask_bg)

    # Morphological closing untuk isi gap kecil
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_teks = cv2.morphologyEx(mask_teks, cv2.MORPH_CLOSE, kernel)

    return mask_teks


def preprocess_plat(crop_bgr):
    """
    Pipeline preprocessing lengkap untuk plat Indonesia.
    Mengembalikan list kandidat gambar (BGR) untuk dikirim ke OCR.
    """
    # 1. Resize 3x
    crop = cv2.resize(crop_bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    # 2. Color segmentation
    mask_teks = color_segment_plat(crop)

    # 3. Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 4. CLAHE — tingkatkan kontras lokal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 5. Denoising
    gray = denoise(gray)

    # 6. Deskewing
    gray = deskew(gray)

    # 7. Morphological close untuk perkuat karakter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # 8. Buat beberapa kandidat threshold
    # Kandidat 1: Adaptive threshold (paling andal untuk cahaya tidak merata)
    thresh_adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)

    # Kandidat 2: Otsu threshold
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Kandidat 3: Gabungkan mask segmentasi warna dengan grayscale
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask_teks)

    # Konversi semua kandidat ke BGR agar bisa masuk PaddleOCR
    kandidat = [
        cv2.cvtColor(thresh_adapt, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(thresh_otsu, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(gray_masked, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),       # fallback plain gray
    ]
    return kandidat


# ============================================================
# OCR + POSTPROCESSING
# ============================================================

def ocr_kandidat(ocr, img_bgr, conf_threshold=0.3):
    """Jalankan PaddleOCR pada satu kandidat gambar, kembalikan list teks."""
    hasil = ocr.predict(img_bgr)
    teks_list = []
    if not hasil or not hasil[0]:
        return teks_list
    rec_texts = hasil[0].get('rec_texts', [])
    rec_scores = hasil[0].get('rec_scores', [])
    for idx, item in enumerate(rec_texts):
        score = rec_scores[idx] if idx < len(rec_scores) else 0
        if score >= conf_threshold:
            cleaned = ''.join(c for c in item.upper()
                              if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')
            if cleaned.strip():
                teks_list.append(cleaned.strip())
    return teks_list


# ============================================================
# KODE WILAYAH RESMI INDONESIA
# ============================================================
KODE_WILAYAH = {
    # Sumatera
    'BL', 'BK', 'BB', 'BA', 'BM', 'BP', 'BG', 'BN', 'BE', 'BD', 'BH',
    # Jawa Barat & Banten
    'A', 'B', 'D', 'E', 'F', 'T', 'Z',
    # Jawa Tengah & DIY
    'G', 'H', 'K', 'R', 'AA', 'AD', 'AB',
    # Jawa Timur
    'L', 'W', 'N', 'S', 'P', 'AG', 'AE', 'M',
    # Bali & Nusa Tenggara
    'DK', 'DR', 'EA', 'DH', 'EB', 'ED',
    # Kalimantan
    'KB', 'DA', 'KH', 'KT', 'KU',
    # Sulawesi
    'DB', 'DL', 'DM', 'DN', 'DT', 'DD', 'DP', 'DC', 'DE', 'DG',
    # Papua & Maluku
    'PA', 'PB',
}

# Koreksi OCR: karakter yang sering salah di posisi huruf vs angka
CHAR_TO_DIGIT = {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'Z': '2', 'G': '6', 'Q': '0'}
DIGIT_TO_CHAR = {'0': 'O', '1': 'I', '8': 'B', '2': 'Z', '6': 'G', '5': 'S'}


def koreksi_prefix(raw_prefix):
    """Koreksi karakter di bagian kode wilayah (harus huruf semua)."""
    result = ""
    for c in raw_prefix:
        if c.isalpha():
            result += c
        elif c in DIGIT_TO_CHAR:
            result += DIGIT_TO_CHAR[c]
    return result


def koreksi_digits(raw_digits):
    """Koreksi karakter di bagian nomor registrasi (harus angka semua)."""
    result = ""
    for c in raw_digits:
        if c.isdigit():
            result += c
        elif c in CHAR_TO_DIGIT:
            result += CHAR_TO_DIGIT[c]
    return result


def koreksi_suffix(raw_suffix):
    """Koreksi karakter di bagian kode sub-wilayah (harus huruf semua)."""
    result = ""
    for c in raw_suffix:
        if c.isalpha():
            result += c
        elif c in DIGIT_TO_CHAR:
            result += DIGIT_TO_CHAR[c]
    return result


def parse_plat(raw):
    """
    Parse string mentah plat menjadi (prefix, digits, suffix).
    Mendukung kode wilayah 1 atau 2 huruf.
    """
    # Regex: coba 2 huruf dulu (kode wilayah 2 huruf seperti BL, AA, DK)
    pattern2 = re.compile(r'^([A-Z]{2})(\d{1,4})([A-Z]{1,3})$')
    # Regex: kode wilayah 1 huruf
    pattern1 = re.compile(r'^([A-Z]{1})(\d{1,4})([A-Z]{1,3})$')

    m = pattern2.match(raw)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = pattern1.match(raw)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None


def format_plat_indonesia(teks_raw):
    """
    Postprocessing: format hasil OCR ke format plat Indonesia resmi.
    Format : [Kode Wilayah] [Nomor Registrasi] [Kode Sub-wilayah]
    Contoh : B 1234 ABC | L 5678 XY | BL 999 VI | AA 1234 AB

    Proses:
    1. Bersihkan & gabungkan token OCR
    2. Koreksi karakter OCR berdasarkan posisi (huruf/angka)
    3. Validasi kode wilayah terhadap daftar resmi
    4. Format output standar
    """
    # Gabungkan semua token, bersihkan karakter non-alfanumerik
    teks = ' '.join(teks_raw).upper().strip()
    teks = re.sub(r'[^A-Z0-9 ]', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()

    if not teks:
        return ""

    raw = ''.join(teks.split())

    # --- Tahap 1: coba parse langsung tanpa koreksi ---
    prefix, digits, suffix = parse_plat(raw)

    # --- Tahap 2: koreksi karakter lalu parse ulang ---
    if not prefix:
        # Ekstraksi manual dengan koreksi
        i = 0
        # Ambil maks 2 huruf di depan
        raw_prefix = ""
        while i < len(raw) and len(raw_prefix) < 2:
            c = raw[i]
            if c.isalpha() or c in DIGIT_TO_CHAR:
                raw_prefix += c
                i += 1
            else:
                break

        # Cek apakah 2 huruf valid sebagai kode wilayah dulu
        # Kalau tidak, coba 1 huruf
        prefix_2 = koreksi_prefix(raw_prefix)
        if len(prefix_2) == 2 and prefix_2 in KODE_WILAYAH:
            prefix_candidate = prefix_2
        elif len(prefix_2) >= 1 and prefix_2[0] in KODE_WILAYAH:
            prefix_candidate = prefix_2[0]
            i -= (len(raw_prefix) - 1)  # kembalikan 1 karakter
        else:
            prefix_candidate = prefix_2  # tetap coba

        # Ambil angka tengah
        raw_digits = ""
        while i < len(raw) and len(raw_digits) < 4:
            c = raw[i]
            if c.isdigit() or c in CHAR_TO_DIGIT:
                raw_digits += c
                i += 1
            else:
                break

        # Ambil huruf belakang
        raw_suffix = ""
        while i < len(raw) and len(raw_suffix) < 3:
            c = raw[i]
            if c.isalpha() or c in DIGIT_TO_CHAR:
                raw_suffix += c
                i += 1
            else:
                break

        prefix = prefix_candidate
        digits = koreksi_digits(raw_digits)
        suffix = koreksi_suffix(raw_suffix)

    else:
        # Koreksi karakter pada hasil parse langsung
        prefix = koreksi_prefix(prefix)
        digits = koreksi_digits(digits)
        suffix = koreksi_suffix(suffix)

    # --- Tahap 3: validasi kode wilayah ---
    if not prefix or not digits or not suffix:
        return teks  # kembalikan teks bersih kalau parsing gagal

    # Cek validasi — coba prefix 2 huruf dulu, lalu 1 huruf
    wilayah_valid = False
    if prefix in KODE_WILAYAH:
        wilayah_valid = True
    elif len(prefix) > 1 and prefix[0] in KODE_WILAYAH:
        # Mungkin karakter kedua salah baca, coba potong
        prefix = prefix[0]
        wilayah_valid = True

    if not wilayah_valid:
        # Tetap kembalikan hasil meski kode wilayah tidak dikenal
        # (bisa plat luar negeri atau OCR belum sempurna)
        return f"{prefix} {digits} {suffix} [?]"

    return f"{prefix} {digits} {suffix}"


def baca_plat_crop(crop_bgr, ocr):
    """Fungsi OCR dari crop langsung — dipanggil oleh OCR worker thread."""
    if crop_bgr is None or crop_bgr.size == 0:
        return ""
    kandidat_list = preprocess_plat(crop_bgr)
    for kandidat in kandidat_list:
        teks_list = ocr_kandidat(ocr, kandidat)
        if teks_list:
            hasil_format = format_plat_indonesia(teks_list)
            if hasil_format:
                return hasil_format
    return ""


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (aA + aB - inter)


class KameraNode(Node):
    def __init__(self):
        super().__init__('kamera_node')
        self.pub = self.create_publisher(String, '/kamera_data', 10)

        self.get_logger().info("Loading YOLO model...")
        self.model = YOLO(MODEL_PATH)

        self.get_logger().info("Loading PaddleOCR...")
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='en'
        )
        self.get_logger().info("Kamera node siap!")

        self.plat_memory = {}
        self.plat_counter = {}
        self.golongan_memory = {}
        self.snapshot_ids = set()
        self.lock = threading.Lock()

        # === THREADING SETUP ===
        # Queue input OCR: (crop_bgr, pbox, best_tid)
        self.ocr_queue = queue.Queue(maxsize=2)
        # Queue hasil OCR: (best_tid, teks)
        self.ocr_result_queue = queue.Queue()
        self._stop_event = threading.Event()

        # Jalankan worker OCR di thread terpisah
        self.ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
        self.ocr_thread.start()
        self.get_logger().info("OCR thread berjalan di background.")

        os.makedirs(SAVE_DIR, exist_ok=True)

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self.get_logger().error("Kamera tidak bisa dibuka!")
            return

        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = f"{SAVE_DIR}/rekaman_{timestamp_file}.mp4"
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(
            self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h)
        )
        self.get_logger().info(f"Rekaman disimpan ke: {self.video_path}")

        self.frame_count = 0
        self.fps = 0.0
        self.t_start = time.time()

        # Timer loop ~33ms = ~30fps
        self.timer = self.create_timer(0.033, self.process_frame)

    def _ocr_worker(self):
        """
        Worker thread: ambil crop dari ocr_queue,
        jalankan OCR, taruh hasil ke ocr_result_queue.
        Berjalan terus di background, tidak blocking main thread.
        """
        while not self._stop_event.is_set():
            try:
                crop_bgr, best_tid = self.ocr_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                teks = baca_plat_crop(crop_bgr, self.ocr)
                if teks and best_tid is not None:
                    self.ocr_result_queue.put((best_tid, teks))
            except Exception as e:
                pass
            finally:
                self.ocr_queue.task_done()

    def _ambil_hasil_ocr(self):
        """Ambil semua hasil OCR yang sudah selesai dari result queue."""
        while not self.ocr_result_queue.empty():
            try:
                best_tid, teks = self.ocr_result_queue.get_nowait()
                with self.lock:
                    if best_tid not in self.plat_counter:
                        self.plat_counter[best_tid] = {}
                    self.plat_counter[best_tid][teks] = \
                        self.plat_counter[best_tid].get(teks, 0) + 1
                    best_teks = max(
                        self.plat_counter[best_tid],
                        key=self.plat_counter[best_tid].get
                    )
                    self.plat_memory[best_tid] = best_teks
            except queue.Empty:
                break

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Frame tidak terbaca")
            return

        # Hitung FPS
        self.frame_count += 1
        elapsed = time.time() - self.t_start
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.t_start = time.time()

        # Ambil hasil OCR yang sudah selesai di background
        self._ambil_hasil_ocr()

        results = self.model.track(
            source=frame,
            tracker="bytetrack.yaml",
            conf=0.5,
            iou=0.3,
            persist=True,
            verbose=False
        )

        if results is None or len(results) == 0:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Kamera MLFF", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown()
            return

        result = results[0]
        boxes = result.boxes

        kendaraan_list = []
        plat_list = []

        for i, cls in enumerate(boxes.cls.int().cpu().numpy()):
            box = boxes.xyxy[i].cpu().numpy()
            if cls in VEHICLE_CLASSES:
                tid = int(boxes.id[i]) if boxes.id is not None else -1
                kendaraan_list.append((tid, int(cls), box))
                with self.lock:
                    self.golongan_memory[tid] = GOLONGAN_MAP.get(int(cls), 2)
            elif cls == PLATE_CLASS:
                plat_list.append(box)

        # Kirim crop plat ke OCR thread (non-blocking)
        for pbox in plat_list:
            best_iou, best_tid = 0, None
            for tid, cls, vbox in kendaraan_list:
                score = iou(pbox, vbox)
                if score > best_iou:
                    best_iou, best_tid = score, tid

            if best_tid is not None:
                x1, y1, x2, y2 = map(int, pbox)
                pad = 17
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad)
                y2 = min(frame.shape[0], y2 + pad)
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size > 0:
                    try:
                        # put_nowait agar tidak blocking frame loop
                        self.ocr_queue.put_nowait((crop, best_tid))
                    except queue.Full:
                        pass  # skip kalau queue penuh, tidak masalah

        # Gambar bounding box pakai hasil OCR terakhir (dari memory)
        for tid, cls, box in kendaraan_list:
            with self.lock:
                plat = self.plat_memory.get(tid, "")
                golongan = self.golongan_memory.get(tid, 2)
            x1, y1, x2, y2 = map(int, box)

            color = (0, 200, 50)
            nama = self.model.names[cls]
            label = f"{nama} ID:{tid}"
            if plat:
                label += f" | {plat}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if plat:
                data = {
                    "track_id": tid,
                    "plat": plat,
                    "golongan": golongan,
                    "timestamp": datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
                    "fps": round(self.fps, 1)
                }
                msg = String()
                msg.data = json.dumps(data)
                self.pub.publish(msg)
                self.get_logger().info(f"Publish: {data}")

                with self.lock:
                    if tid not in self.snapshot_ids:
                        snap_path = f"{SAVE_DIR}/snapshot_{tid}_{plat.replace(' ', '_')}_{datetime.now().strftime('%H%M%S')}.jpg"
                        cv2.imwrite(snap_path, frame)
                        self.get_logger().info(f"Snapshot disimpan: {snap_path}")
                        self.snapshot_ids.add(tid)

        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.out.write(frame)
        cv2.imshow("Kamera MLFF", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.shutdown()
        elif key == ord('s'):
            snap_path = f"{SAVE_DIR}/manual_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(snap_path, frame)
            self.get_logger().info(f"Snapshot manual: {snap_path}")

    def shutdown(self):
        self._stop_event.set()
        self.ocr_thread.join(timeout=2.0)
        self.cap.release()
        self.out.release()
        self.get_logger().info(f"Rekaman selesai disimpan: {self.video_path}")
        cv2.destroyAllWindows()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = KameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()


if __name__ == '__main__':
    main()