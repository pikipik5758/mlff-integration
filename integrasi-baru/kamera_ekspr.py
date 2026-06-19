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

# === KONFIGURASI COUNTING LINE (garis vertikal) ===
# X koordinat garis virtual (piksel dari kiri frame)
# Contoh: 320 = tengah frame resolusi 640px
# Ubah sesuai kebutuhan!
COUNTING_LINE_X = 320

# Toleransi zona crossing (piksel kiri-kanan dari garis)
# Kendaraan dihitung saat center-x melewati zona ini
COUNTING_ZONE = 10

# Arah counting: "left", "right", atau "both"
COUNTING_DIRECTION = "both"


# ============================================================
# PREPROCESSING UTILITIES
# ============================================================

def deskew(gray):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 1.0:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def denoise(gray):
    return cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)


def color_segment_plat(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask_putih  = cv2.inRange(hsv, (0,   0,   160), (180, 40,  255))
    mask_kuning = cv2.inRange(hsv, (15,  80,  80),  (35,  255, 255))
    mask_merah1 = cv2.inRange(hsv, (0,   80,  80),  (10,  255, 255))
    mask_merah2 = cv2.inRange(hsv, (165, 80,  80),  (180, 255, 255))
    mask_merah  = cv2.bitwise_or(mask_merah1, mask_merah2)
    mask_hijau  = cv2.inRange(hsv, (40,  50,  50),  (85,  255, 255))
    mask_bg = cv2.bitwise_or(mask_putih, mask_kuning)
    mask_bg = cv2.bitwise_or(mask_bg, mask_merah)
    mask_bg = cv2.bitwise_or(mask_bg, mask_hijau)
    mask_teks = cv2.bitwise_not(mask_bg)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(mask_teks, cv2.MORPH_CLOSE, kernel)


def preprocess_plat(crop_bgr):
    crop = cv2.resize(crop_bgr, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    mask_teks = color_segment_plat(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = denoise(gray)
    gray = deskew(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask_teks)
    return [
        cv2.cvtColor(thresh_adapt, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(thresh_otsu,  cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(gray_masked,  cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(gray,         cv2.COLOR_GRAY2BGR),
    ]


# ============================================================
# OCR + POSTPROCESSING
# ============================================================

def ocr_kandidat(ocr, img_bgr, conf_threshold=0.2):
    hasil = ocr.predict(img_bgr)
    teks_list = []
    if not hasil or not hasil[0]:
        return teks_list
    rec_texts  = hasil[0].get('rec_texts', [])
    rec_scores = hasil[0].get('rec_scores', [])
    for idx, item in enumerate(rec_texts):
        score = rec_scores[idx] if idx < len(rec_scores) else 0
        if score >= conf_threshold:
            cleaned = ''.join(c for c in item.upper() if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')
            if cleaned.strip():
                teks_list.append(cleaned.strip())
    return teks_list


# ============================================================
# KODE WILAYAH RESMI INDONESIA
# ============================================================
KODE_WILAYAH = {
    'BL','BK','BB','BA','BM','BP','BG','BN','BE','BD','BH',
    'A','B','D','E','F','T','Z',
    'G','H','K','R','AA','AD','AB',
    'L','W','N','S','P','AG','AE','M',
    'DK','DR','EA','DH','EB','ED',
    'KB','DA','KH','KT','KU',
    'DB','DL','DM','DN','DT','DD','DP','DC','DE','DG',
    'PA','PB',
}

CHAR_TO_DIGIT = {'O':'0','I':'1','S':'5','B':'8','Z':'2','G':'6','Q':'0'}
DIGIT_TO_CHAR = {'0':'O','1':'I','8':'B','2':'Z','6':'G','5':'S'}

def koreksi_prefix(s):
    return ''.join(c if c.isalpha() else DIGIT_TO_CHAR.get(c,'') for c in s)

def koreksi_digits(s):
    return ''.join(c if c.isdigit() else CHAR_TO_DIGIT.get(c,'') for c in s)

def koreksi_suffix(s):
    return ''.join(c if c.isalpha() else DIGIT_TO_CHAR.get(c,'') for c in s)

def parse_plat(raw):
    for pat in [r'^([A-Z]{2})(\d{1,4})([A-Z]{1,3})$', r'^([A-Z]{1})(\d{1,4})([A-Z]{1,3})$']:
        m = re.compile(pat).match(raw)
        if m:
            return m.group(1), m.group(2), m.group(3)
    return None, None, None

def format_plat_indonesia(teks_raw):
    teks = ' '.join(teks_raw).upper().strip()
    teks = re.sub(r'[^A-Z0-9 ]', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    if not teks:
        return ""
    raw = ''.join(teks.split())

    prefix, digits, suffix = parse_plat(raw)

    if not prefix:
        i = 0
        raw_prefix = ""
        while i < len(raw) and len(raw_prefix) < 2:
            c = raw[i]
            if c.isalpha() or c in DIGIT_TO_CHAR:
                raw_prefix += c; i += 1
            else:
                break
        prefix_2 = koreksi_prefix(raw_prefix)
        if len(prefix_2) == 2 and prefix_2 in KODE_WILAYAH:
            prefix_candidate = prefix_2
        elif len(prefix_2) >= 1 and prefix_2[0] in KODE_WILAYAH:
            prefix_candidate = prefix_2[0]
            i -= (len(raw_prefix) - 1)
        else:
            prefix_candidate = prefix_2

        raw_digits = ""
        while i < len(raw) and len(raw_digits) < 4:
            c = raw[i]
            if c.isdigit() or c in CHAR_TO_DIGIT:
                raw_digits += c; i += 1
            else:
                break

        raw_suffix = ""
        while i < len(raw) and len(raw_suffix) < 3:
            c = raw[i]
            if c.isalpha() or c in DIGIT_TO_CHAR:
                raw_suffix += c; i += 1
            else:
                break

        prefix = prefix_candidate
        digits = koreksi_digits(raw_digits)
        suffix = koreksi_suffix(raw_suffix)
    else:
        prefix = koreksi_prefix(prefix)
        digits = koreksi_digits(digits)
        suffix = koreksi_suffix(suffix)

    if not prefix or not digits or not suffix:
        return teks

    wilayah_valid = prefix in KODE_WILAYAH
    if not wilayah_valid and len(prefix) > 1 and prefix[0] in KODE_WILAYAH:
        prefix = prefix[0]
        wilayah_valid = True

    if not wilayah_valid:
        return f"{prefix} {digits} {suffix} [?]"
    return f"{prefix} {digits} {suffix}"


def baca_plat_crop(crop_bgr, ocr):
    if crop_bgr is None or crop_bgr.size == 0:
        return ""
    for kandidat in preprocess_plat(crop_bgr):
        teks_list = ocr_kandidat(ocr, kandidat)
        if teks_list:
            hasil = format_plat_indonesia(teks_list)
            if hasil:
                return hasil
    return ""


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (aA + aB - inter)


# ============================================================
# COUNTING UTILITIES
# ============================================================

def get_center_x(box):
    """Ambil center-x dari bounding box [x1,y1,x2,y2]."""
    return int((box[0] + box[2]) / 2)

def get_center_y(box):
    """Ambil center-y dari bounding box."""
    return int((box[1] + box[3]) / 2)

def cek_crossing(prev_cx, curr_cx, line_x, zone, direction):
    """
    Cek apakah kendaraan baru saja melewati garis vertikal.
    Mengembalikan arah crossing: 'right', 'left', atau None.
    """
    # Kendaraan dianggap crossing jika center melewati zona garis
    if prev_cx is None:
        return None
    crossed_right = prev_cx < (line_x - zone) and curr_cx >= (line_x - zone)
    crossed_left  = prev_cx > (line_x + zone) and curr_cx <= (line_x + zone)

    if direction == "right" and crossed_right:
        return "right"
    elif direction == "left" and crossed_left:
        return "left"
    elif direction == "both":
        if crossed_right:
            return "right"
        if crossed_left:
            return "left"
    return None

def draw_counting_overlay(frame, line_x, count_total, count_per_golongan, count_per_id):
    """Gambar garis virtual dan overlay info counting di frame."""
    h, w = frame.shape[:2]

    # === Garis vertikal ===
    cv2.line(frame, (line_x, 0), (line_x, h), (0, 0, 255), 2)
    cv2.putText(frame, "COUNTING LINE", (line_x + 5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # === Warna & label per golongan ===
    GOL_COLOR = {
        1: (0,   255, 255),
        2: (0,   200, 255),
        3: (0,   255, 128),
        4: (255, 128, 0),
        5: (128, 0,   255),
    }
    GOL_LABEL = {
        1: "Gol 1 - Motor",
        2: "Gol 2 - Sedan",
        3: "Gol 3 - Minibus",
        4: "Gol 4 - Bus/Truk",
        5: "Gol 5 - Truk Besar",
    }

    # === Panel utama pojok kanan atas ===
    panel_w = 230
    panel_h = 30 + 8 + 30 * 5 + 10
    panel_x = w - panel_w - 10
    panel_y = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 8, panel_y - 8),
                  (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (panel_x - 8, panel_y - 8),
                  (panel_x + panel_w, panel_y + panel_h), (180, 180, 180), 1)

    cv2.putText(frame, "VEHICLE COUNTER", (panel_x, panel_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"TOTAL : {count_total}", (panel_x, panel_y + 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 255), 2)

    sep_y = panel_y + 48
    cv2.line(frame, (panel_x - 8, sep_y), (panel_x + panel_w, sep_y), (120, 120, 120), 1)

    for i, gol in enumerate([1, 2, 3, 4, 5]):
        jml   = count_per_golongan.get(gol, 0)
        color = GOL_COLOR[gol]
        label = GOL_LABEL[gol]
        row_y = sep_y + 10 + i * 30
        cv2.rectangle(frame, (panel_x, row_y - 10), (panel_x + 10, row_y + 4), color, -1)
        cv2.putText(frame, label, (panel_x + 16, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
        jml_str = str(jml)
        (jw, _), _ = cv2.getTextSize(jml_str, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.putText(frame, jml_str, (panel_x + panel_w - jw - 4, row_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)

    # === Bar chart distribusi di bagian bawah ===
    bar_y     = h - 45
    bar_x     = 10
    bar_h     = 30
    bar_total = w - 20

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (bar_x, bar_y - 22), (bar_x + bar_total, bar_y + bar_h + 5),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "DISTRIBUSI GOLONGAN", (bar_x, bar_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if count_total > 0:
        x_cursor = bar_x
        for gol in [1, 2, 3, 4, 5]:
            jml   = count_per_golongan.get(gol, 0)
            color = GOL_COLOR[gol]
            seg_w = int((jml / count_total) * bar_total)
            if seg_w > 0:
                cv2.rectangle(frame, (x_cursor, bar_y),
                              (x_cursor + seg_w, bar_y + bar_h), color, -1)
                if seg_w > 35:
                    pct = int(jml / count_total * 100)
                    cv2.putText(frame, f"{pct}%", (x_cursor + 4, bar_y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                x_cursor += seg_w
    else:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_total, bar_y + bar_h),
                      (60, 60, 60), -1)
        cv2.putText(frame, "Belum ada kendaraan", (bar_x + 10, bar_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    return frame


# ============================================================
# KAMERA NODE
# ============================================================

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

        self.plat_memory     = {}
        self.plat_counter    = {}
        self.golongan_memory = {}
        self.snapshot_ids    = set()
        self.lock            = threading.Lock()

        # === COUNTING STATE ===
        self.prev_cx         = {}   # track_id -> center_x frame sebelumnya
        self.counted_ids     = set()  # track_id yang sudah dihitung (unik)
        self.count_total     = 0
        self.count_per_golongan = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.count_per_id    = {}   # track_id -> info saat dihitung

        # === THREADING OCR ===
        self.ocr_queue        = queue.Queue(maxsize=2)
        self.ocr_result_queue = queue.Queue()
        self._stop_event      = threading.Event()
        self.ocr_thread       = threading.Thread(target=self._ocr_worker, daemon=True)
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
        self.fps         = 0.0
        self.t_start     = time.time()
        self.timer       = self.create_timer(0.033, self.process_frame)

    # ----------------------------------------------------------
    # OCR THREAD
    # ----------------------------------------------------------

    def _ocr_worker(self):
        while not self._stop_event.is_set():
            try:
                crop_bgr, best_tid = self.ocr_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                teks = baca_plat_crop(crop_bgr, self.ocr)
                if teks and best_tid is not None:
                    self.ocr_result_queue.put((best_tid, teks))
            except Exception:
                pass
            finally:
                self.ocr_queue.task_done()

    def _ambil_hasil_ocr(self):
        while not self.ocr_result_queue.empty():
            try:
                best_tid, teks = self.ocr_result_queue.get_nowait()
                with self.lock:
                    if best_tid not in self.plat_counter:
                        self.plat_counter[best_tid] = {}
                    self.plat_counter[best_tid][teks] = \
                        self.plat_counter[best_tid].get(teks, 0) + 1
                    self.plat_memory[best_tid] = max(
                        self.plat_counter[best_tid],
                        key=self.plat_counter[best_tid].get
                    )
            except queue.Empty:
                break

    # ----------------------------------------------------------
    # COUNTING
    # ----------------------------------------------------------

    def _update_counting(self, tid, box, golongan):
        """
        Cek crossing garis untuk track_id ini.
        Kendaraan hanya dihitung 1x per track_id (id unik).
        """
        curr_cx = get_center_x(box)
        prev_cx = self.prev_cx.get(tid)

        arah = cek_crossing(prev_cx, curr_cx, COUNTING_LINE_X, COUNTING_ZONE, COUNTING_DIRECTION)

        self.prev_cx[tid] = curr_cx  # update posisi sebelumnya

        if arah and tid not in self.counted_ids:
            self.counted_ids.add(tid)
            self.count_total += 1
            self.count_per_golongan[golongan] = self.count_per_golongan.get(golongan, 0) + 1
            self.count_per_id[tid] = {
                "golongan": golongan,
                "arah": arah,
                "plat": self.plat_memory.get(tid, ""),
                "waktu": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
            }
            self.get_logger().info(
                f"[COUNTING] ID:{tid} | Gol:{golongan} | Arah:{arah} | "
                f"Total:{self.count_total} | Plat:{self.plat_memory.get(tid,'')}"
            )

    # ----------------------------------------------------------
    # MAIN FRAME LOOP
    # ----------------------------------------------------------

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Frame tidak terbaca")
            return

        self.frame_count += 1
        elapsed = time.time() - self.t_start
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.t_start = time.time()

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
            draw_counting_overlay(frame, COUNTING_LINE_X,
                                  self.count_total, self.count_per_golongan, self.count_per_id)
            cv2.imshow("Kamera MLFF", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown()
            return

        result  = results[0]
        boxes   = result.boxes
        kendaraan_list = []
        plat_list      = []

        for i, cls in enumerate(boxes.cls.int().cpu().numpy()):
            box = boxes.xyxy[i].cpu().numpy()
            if cls in VEHICLE_CLASSES:
                tid = int(boxes.id[i]) if boxes.id is not None else -1
                kendaraan_list.append((tid, int(cls), box))
                with self.lock:
                    self.golongan_memory[tid] = GOLONGAN_MAP.get(int(cls), 2)
            elif cls == PLATE_CLASS:
                plat_list.append(box)

        # Kirim crop ke OCR thread
        for pbox in plat_list:
            best_iou, best_tid = 0, None
            for tid, cls, vbox in kendaraan_list:
                score = iou(pbox, vbox)
                if score > best_iou:
                    best_iou, best_tid = score, tid
            if best_tid is not None:
                x1, y1, x2, y2 = map(int, pbox)
                pad = 15
                x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad); y2 = min(frame.shape[0], y2 + pad)
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size > 0:
                    try:
                        self.ocr_queue.put_nowait((crop, best_tid))
                    except queue.Full:
                        pass

        # Gambar bbox + update counting
        for tid, cls, box in kendaraan_list:
            with self.lock:
                plat     = self.plat_memory.get(tid, "")
                golongan = self.golongan_memory.get(tid, 2)

            # Update counting (cek crossing garis)
            self._update_counting(tid, box, golongan)

            x1, y1, x2, y2 = map(int, box)
            cx = get_center_x(box)
            cy = get_center_y(box)

            # Warna berbeda kalau sudah dihitung
            color = (0, 255, 128) if tid in self.counted_ids else (0, 200, 50)
            nama  = self.model.names[cls]
            label = f"{nama} ID:{tid}"
            if plat:
                label += f" | {plat}"
            if tid in self.counted_ids:
                label += " ✓"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Titik center kendaraan
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            if plat:
                data = {
                    "track_id": tid,
                    "plat":     plat,
                    "golongan": golongan,
                    "counted":  tid in self.counted_ids,
                    "count_total": self.count_total,
                    "timestamp": datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
                    "fps":      round(self.fps, 1)
                }
                msg = String()
                msg.data = json.dumps(data)
                self.pub.publish(msg)
                self.get_logger().info(f"Publish: {data}")

                with self.lock:
                    if tid not in self.snapshot_ids:
                        snap_path = (f"{SAVE_DIR}/snapshot_{tid}_"
                                     f"{plat.replace(' ','_')}_"
                                     f"{datetime.now().strftime('%H%M%S')}.jpg")
                        cv2.imwrite(snap_path, frame)
                        self.get_logger().info(f"Snapshot disimpan: {snap_path}")
                        self.snapshot_ids.add(tid)

        # Overlay FPS + counting
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        draw_counting_overlay(frame, COUNTING_LINE_X,
                              self.count_total, self.count_per_golongan, self.count_per_id)

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
        # Print rekap counting
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"REKAP COUNTING — Total: {self.count_total}")
        for gol, jml in self.count_per_golongan.items():
            self.get_logger().info(f"  Golongan {gol}: {jml} kendaraan")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Rekaman disimpan: {self.video_path}")
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