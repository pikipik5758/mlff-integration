#!/usr/bin/env python3
"""
MLFF Vehicle Counting - Standalone Testing (Tanpa ROS 2)
=========================================================
Cara pakai:
  python mlff_counting_test.py --video path/ke/video.mp4

Langkah interaktif:
  1. Frame pertama ditampilkan
  2. Klik 2 titik (kiri→kanan) untuk garis LANE KIRI
  3. Klik 2 titik (kiri→kanan) untuk garis LANE KANAN
  4. Tekan ENTER untuk mulai counting
  5. Tekan Q saat video berjalan untuk berhenti
"""

import cv2
import numpy as np
import json
import time
import os
import re
import threading
import queue
import argparse
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ============================================================
# KONFIGURASI — sesuaikan path model
# ============================================================
MODEL_PATH  = "/home/piki/PA/integrasi-baru/tes-video/best0895.pt"
SAVE_DIR    = "./hasil_test"

VEHICLE_CLASSES = {0, 1, 2, 3, 4}
PLATE_CLASS     = 5

GOLONGAN_MAP = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

GOLONGAN_LABEL = {
    1: "Golongan I",
    2: "Golongan II",
    3: "Golongan III",
    4: "Golongan IV",
    5: "Golongan V",
}

LANE_COLOR = {
    "kiri":  (0, 255, 255),   # kuning
    "kanan": (255, 100, 255), # pink/magenta
}

GOL_COLOR = {
    1: (0,   255, 255),
    2: (0,   200, 255),
    3: (0,   255, 128),
    4: (255, 128, 0),
    5: (128, 0,   255),
}


# ============================================================
# PREPROCESSING + OCR (sama seperti kode asli)
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
    thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 15, 8)
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask_teks)
    return [
        cv2.cvtColor(thresh_adapt, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(thresh_otsu,  cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(gray_masked,  cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(gray,         cv2.COLOR_GRAY2BGR),
    ]


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
    return ''.join(c if c.isalpha() else DIGIT_TO_CHAR.get(c, '') for c in s)

def koreksi_digits(s):
    return ''.join(c if c.isdigit() else CHAR_TO_DIGIT.get(c, '') for c in s)

def koreksi_suffix(s):
    return ''.join(c if c.isalpha() else DIGIT_TO_CHAR.get(c, '') for c in s)

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
# GARIS COUNTING INTERAKTIF
# ============================================================

class LineSelector:
    """
    Tampilkan frame, minta user klik 2 titik per lane.
    Urutan: Lane KIRI dulu (2 klik), lalu Lane KANAN (2 klik).
    """
    def __init__(self, frame):
        self.frame_orig = frame.copy()
        self.frame_disp = frame.copy()
        self.points     = []   # list of (x, y)
        self.lanes      = {}   # "kiri" -> (pt1, pt2), "kanan" -> (pt1, pt2)
        self.step       = 0    # 0=kiri pt1, 1=kiri pt2, 2=kanan pt1, 3=kanan pt2
        self.done       = False

        self.STEP_INFO = [
            ("LANE KIRI",  "Klik titik KIRI garis Lane Kiri",  LANE_COLOR["kiri"]),
            ("LANE KIRI",  "Klik titik KANAN garis Lane Kiri", LANE_COLOR["kiri"]),
            ("LANE KANAN", "Klik titik KIRI garis Lane Kanan", LANE_COLOR["kanan"]),
            ("LANE KANAN", "Klik titik KANAN garis Lane Kanan",LANE_COLOR["kanan"]),
        ]

    def mouse_callback(self, event, x, y, flags, param):
        if self.done:
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # Preview garis sementara
            self._redraw()
            cv2.circle(self.frame_disp, (x, y), 6,
                       self.STEP_INFO[self.step][2], -1)
            if self.step in [1, 3] and self.points:
                cv2.line(self.frame_disp, self.points[-1], (x, y),
                         self.STEP_INFO[self.step][2], 2)

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.step += 1

            if self.step == 2:
                self.lanes["kiri"] = (self.points[0], self.points[1])
            elif self.step == 4:
                self.lanes["kanan"] = (self.points[2], self.points[3])
                self.done = True

            self._redraw()

    def _redraw(self):
        self.frame_disp = self.frame_orig.copy()

        # Gambar garis yang sudah selesai
        if "kiri" in self.lanes:
            cv2.line(self.frame_disp,
                     self.lanes["kiri"][0], self.lanes["kiri"][1],
                     LANE_COLOR["kiri"], 3)
            self._label_lane(self.frame_disp, self.lanes["kiri"], "LANE KIRI", LANE_COLOR["kiri"])

        if "kanan" in self.lanes:
            cv2.line(self.frame_disp,
                     self.lanes["kanan"][0], self.lanes["kanan"][1],
                     LANE_COLOR["kanan"], 3)
            self._label_lane(self.frame_disp, self.lanes["kanan"], "LANE KANAN", LANE_COLOR["kanan"])

        # Titik yang sudah diklik
        for pt in self.points:
            cv2.circle(self.frame_disp, pt, 7, (255, 255, 255), -1)
            cv2.circle(self.frame_disp, pt, 5, (0, 0, 0), -1)

        # Instruksi di layar
        if not self.done:
            lane_name, instruksi, color = self.STEP_INFO[self.step]
            overlay = self.frame_disp.copy()
            cv2.rectangle(overlay, (0, 0), (self.frame_disp.shape[1], 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, self.frame_disp, 0.4, 0, self.frame_disp)
            cv2.putText(self.frame_disp,
                        f"[{lane_name}] {instruksi}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(self.frame_disp,
                        f"Klik {self.step + 1}/4",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        else:
            overlay = self.frame_disp.copy()
            cv2.rectangle(overlay, (0, 0), (self.frame_disp.shape[1], 60), (0, 80, 0), -1)
            cv2.addWeighted(overlay, 0.7, self.frame_disp, 0.3, 0, self.frame_disp)
            cv2.putText(self.frame_disp,
                        "Garis selesai! Tekan ENTER untuk mulai counting...",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 100), 2)

    def _label_lane(self, frame, line, label, color):
        mid_x = (line[0][0] + line[1][0]) // 2
        mid_y = (line[0][1] + line[1][1]) // 2
        cv2.putText(frame, label, (mid_x - 50, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    def run(self):
        win = "Setup Garis Counting - Klik 4 titik, lalu ENTER"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        self._redraw()
        cv2.imshow(win, self.frame_disp)
        cv2.waitKey(100)
        cv2.setMouseCallback(win, self.mouse_callback)

        while True:
            cv2.imshow(win, self.frame_disp)
            key = cv2.waitKey(20) & 0xFF
            if key == 13 and self.done:   # ENTER
                break
            elif key == ord('r'):         # Reset
                self.points = []
                self.lanes  = {}
                self.step   = 0
                self.done   = False
                self._redraw()
            elif key == 27:              # ESC = batal
                cv2.destroyWindow(win)
                return None, None

        cv2.destroyWindow(win)
        return self.lanes["kiri"], self.lanes["kanan"]


# ============================================================
# CROSSING LOGIC — GARIS HORIZONTAL (2 TITIK)
# ============================================================

def point_side_of_line(pt, line_p1, line_p2):
    """
    Tentukan sisi titik relatif terhadap garis (p1→p2).
    Positif = satu sisi, Negatif = sisi lain.
    """
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]
    return (pt[0] - line_p1[0]) * dy - (pt[1] - line_p1[1]) * dx


def get_center(box):
    cx = int((box[0] + box[2]) / 2)
    cy = int(box[1] + (box[3] - box[1]) * 0.85)
    return cx, cy


def cek_crossing_line(prev_pt, curr_pt, line_p1, line_p2):
    """
    Cek apakah objek melewati garis dari prev_pt ke curr_pt.
    Return: 'cross' atau None
    """
    if prev_pt is None:
        return None
    side_prev = point_side_of_line(prev_pt, line_p1, line_p2)
    side_curr = point_side_of_line(curr_pt, line_p1, line_p2)
    if side_prev * side_curr < 0:   # berbeda sisi → crossing
        return "cross"
    return None


def assign_lane(cx, cy, lane_kiri, lane_kanan, frame_w):
    """
    Tentukan kendaraan di lane kiri atau kanan berdasarkan posisi center-x.
    Pakai titik tengah antara kedua garis sebagai pembatas.
    """
    # Ambil mid-x dari masing-masing garis
    mid_kiri  = (lane_kiri[0][0]  + lane_kiri[1][0])  // 2
    mid_kanan = (lane_kanan[0][0] + lane_kanan[1][0]) // 2
    batas_x   = (mid_kiri + mid_kanan) // 2
    return "kiri" if cx < batas_x else "kanan"


# ============================================================
# OVERLAY VISUALISASI
# ============================================================

def draw_overlay(frame, lane_kiri, lane_kanan,
                 count_total, count_per_lane, count_per_golongan_per_lane,
                 fps, frame_idx, total_frames):
    h, w = frame.shape[:2]

    # === Gambar garis counting ===
    for lane_name, line in [("KIRI", lane_kiri), ("KANAN", lane_kanan)]:
        color = LANE_COLOR["kiri"] if lane_name == "KIRI" else LANE_COLOR["kanan"]
        cv2.line(frame, line[0], line[1], color, 3)
        mid_x = (line[0][0] + line[1][0]) // 2
        mid_y = (line[0][1] + line[1][1]) // 2
        cv2.putText(frame, f"LANE {lane_name}", (mid_x - 50, mid_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # === Panel kanan atas ===
    panel_w = 260
    row_h   = 28
    n_gol   = 5
    panel_h = 20 + 30 + 10 + row_h * n_gol + 20 + row_h * n_gol + 15
    px = w - panel_w - 10
    py = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (px - 8, py - 8), (px + panel_w, py + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (px - 8, py - 8), (px + panel_w, py + panel_h), (160, 160, 160), 1)

    y = py + 16
    cv2.putText(frame, "VEHICLE COUNTER", (px, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 26
    cv2.putText(frame, f"TOTAL: {count_total}",
                (px, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

    for lane_name in ["kiri", "kanan"]:
        y += 16
        color = LANE_COLOR[lane_name]
        label = f"--- LANE {'KIRI' if lane_name=='kiri' else 'KANAN'} ({count_per_lane.get(lane_name, 0)}) ---"
        cv2.putText(frame, label, (px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        gol_dict = count_per_golongan_per_lane.get(lane_name, {})
        for gol in range(1, 6):
            y += row_h - 4
            jml = gol_dict.get(gol, 0)
            gc  = GOL_COLOR[gol]
            cv2.rectangle(frame, (px, y - 10), (px + 10, y + 4), gc, -1)
            cv2.putText(frame, GOLONGAN_LABEL[gol], (px + 14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            jml_str = str(jml)
            (jw, _), _ = cv2.getTextSize(jml_str, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            cv2.putText(frame, jml_str, (px + panel_w - jw - 4, y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, gc, 1)

    # === FPS + Progress ===
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if total_frames > 0:
        pct = int(frame_idx / total_frames * 100)
        bar_w = 200
        bar_x, bar_y = 10, h - 20
        cv2.rectangle(frame, (bar_x, bar_y - 10), (bar_x + bar_w, bar_y + 4), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y - 10),
                      (bar_x + int(bar_w * pct / 100), bar_y + 4), (0, 200, 100), -1)
        cv2.putText(frame, f"{pct}% ({frame_idx}/{total_frames})",
                    (bar_x + bar_w + 8, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    return frame


# ============================================================
# MAIN PROCESSOR
# ============================================================

class VideoProcessor:
    def __init__(self, video_path, lane_kiri, lane_kanan):
        self.video_path  = video_path
        self.lane_kiri   = lane_kiri    # (pt1, pt2)
        self.lane_kanan  = lane_kanan

        print("Loading YOLO model...")
        self.model = YOLO(MODEL_PATH)

        print("Loading PaddleOCR...")
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='en'
        )

        # State tracking
        self.plat_memory     = {}
        self.plat_counter    = {}
        self.golongan_memory = {}
        self.snapshot_ids    = set()
        self.lock            = threading.Lock()

        # Counting state
        self.prev_center     = {}   # track_id -> (cx, cy) frame sebelumnya
        self.counted_ids     = set()
        self.count_total     = 0
        self.count_per_lane  = {"kiri": 0, "kanan": 0}
        self.count_per_golongan_per_lane = {
            "kiri":  {g: 0 for g in range(1, 6)},
            "kanan": {g: 0 for g in range(1, 6)},
        }

        # OCR threading
        self.ocr_queue        = queue.Queue(maxsize=4)
        self.ocr_result_queue = queue.Queue()
        self._stop_event      = threading.Event()
        self.ocr_thread       = threading.Thread(target=self._ocr_worker, daemon=True)
        self.ocr_thread.start()

        os.makedirs(SAVE_DIR, exist_ok=True)

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

    def _update_counting(self, tid, box, golongan, frame_w):
        cx, cy = get_center(box)
        prev   = self.prev_center.get(tid)
        self.prev_center[tid] = (cx, cy)

        if tid in self.counted_ids:
            return

        # Tentukan lane kendaraan
        lane = assign_lane(cx, cy, self.lane_kiri, self.lane_kanan, frame_w)

        # Cek crossing sesuai lane
        line = self.lane_kiri if lane == "kiri" else self.lane_kanan
        arah = cek_crossing_line(prev, (cx, cy), line[0], line[1])

        if arah:
            self.counted_ids.add(tid)
            self.count_total += 1
            self.count_per_lane[lane] += 1
            self.count_per_golongan_per_lane[lane][golongan] = \
                self.count_per_golongan_per_lane[lane].get(golongan, 0) + 1

            plat = self.plat_memory.get(tid, "")
            waktu = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
            print(f"[COUNT] ID:{tid} | Lane:{lane.upper()} | Gol:{golongan} "
                  f"({GOLONGAN_LABEL[golongan]}) | Total:{self.count_total} | Plat:{plat} | {waktu}")

            # Simpan ke JSON log
            log_entry = {
                "track_id": tid,
                "lane":     lane,
                "golongan": golongan,
                "label":    GOLONGAN_LABEL[golongan],
                "plat":     plat,
                "waktu":    waktu,
                "count_total": self.count_total,
            }
            log_path = os.path.join(SAVE_DIR, "log_counting.jsonl")
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

    # ----------------------------------------------------------

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"ERROR: Tidak bisa membuka video: {self.video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25

        # Output video
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(SAVE_DIR, f"hasil_{ts}.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps_video, (fw, fh))
        print(f"Output video: {out_path}")
        print("Tekan Q untuk berhenti, S untuk screenshot manual")

        frame_idx = 0
        t_start   = time.time()
        fps_disp  = 0.0
        fc_timer  = 0

        cv2.namedWindow("MLFF Counting Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MLFF Counting Test", 1280, 720)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video selesai.")
                break

            frame_idx += 1
            fc_timer  += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps_disp  = fc_timer / elapsed
                fc_timer  = 0
                t_start   = time.time()

            self._ambil_hasil_ocr()

            results = self.model.track(
                source=frame,
                tracker="bytetrack.yaml",
                conf=0.5,
                iou=0.3,
                persist=True,
                verbose=False
            )

            kendaraan_list = []
            plat_list      = []

            if results and len(results) > 0:
                boxes = results[0].boxes
                for i, cls in enumerate(boxes.cls.int().cpu().numpy()):
                    box = boxes.xyxy[i].cpu().numpy()
                    if cls in VEHICLE_CLASSES:
                        tid = int(boxes.id[i]) if boxes.id is not None else -1
                        kendaraan_list.append((tid, int(cls), box))
                        with self.lock:
                            self.golongan_memory[tid] = GOLONGAN_MAP.get(int(cls), 2)
                    elif cls == PLATE_CLASS:
                        plat_list.append(box)

                # OCR plates
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
                        x2 = min(fw, x2 + pad); y2 = min(fh, y2 + pad)
                        crop = frame[y1:y2, x1:x2].copy()
                        if crop.size > 0:
                            try:
                                self.ocr_queue.put_nowait((crop, best_tid))
                            except queue.Full:
                                pass

                # Gambar bbox + counting
                for tid, cls, box in kendaraan_list:
                    with self.lock:
                        plat     = self.plat_memory.get(tid, "")
                        golongan = self.golongan_memory.get(tid, 2)

                    self._update_counting(tid, box, golongan, fw)

                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = get_center(box)

                    lane  = assign_lane(cx, cy, self.lane_kiri, self.lane_kanan, fw)
                    color = LANE_COLOR[lane] if tid in self.counted_ids else (100, 100, 100)
                    nama  = self.model.names[cls]
                    label = f"{nama} ID:{tid}"
                    if plat:
                        label += f" | {plat}"
                    if tid in self.counted_ids:
                        label += " ✓"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                    # Snapshot per kendaraan (saat plat terdeteksi)
                    if plat:
                        with self.lock:
                            if tid not in self.snapshot_ids:
                                snap = (f"{SAVE_DIR}/snap_{tid}_"
                                        f"{plat.replace(' ','_')}_"
                                        f"{datetime.now().strftime('%H%M%S')}.jpg")
                                cv2.imwrite(snap, frame)
                                self.snapshot_ids.add(tid)

            draw_overlay(frame, self.lane_kiri, self.lane_kanan,
                         self.count_total, self.count_per_lane,
                         self.count_per_golongan_per_lane,
                         fps_disp, frame_idx, total_frames)

            out.write(frame)
            cv2.imshow("MLFF Counting Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Dihentikan oleh user.")
                break
            elif key == ord('s'):
                snap = os.path.join(SAVE_DIR, f"manual_{datetime.now().strftime('%H%M%S')}.jpg")
                cv2.imwrite(snap, frame)
                print(f"Screenshot: {snap}")

        cap.release()
        out.release()
        self._stop_event.set()
        cv2.destroyAllWindows()
        self._print_rekap()

    def _print_rekap(self):
        print("\n" + "=" * 55)
        print(f"  REKAP COUNTING — TOTAL: {self.count_total} kendaraan")
        print("=" * 55)
        for lane in ["kiri", "kanan"]:
            print(f"\n  LANE {lane.upper()} ({self.count_per_lane[lane]} kendaraan):")
            for gol in range(1, 6):
                jml = self.count_per_golongan_per_lane[lane].get(gol, 0)
                if jml:
                    print(f"    Golongan {gol} ({GOLONGAN_LABEL[gol]}): {jml}")
        print(f"\n  Log tersimpan di: {SAVE_DIR}/log_counting.jsonl")
        print("=" * 55)


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MLFF Vehicle Counting — Standalone Test (2 Lane, Garis Interaktif)"
    )
    parser.add_argument("--video", required=True, help="Path ke file video (.mp4 / .avi)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: File tidak ditemukan: {args.video}")
        return

    # Ambil frame pertama untuk setup garis
    cap = cv2.VideoCapture(args.video)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Tidak bisa membaca frame pertama video.")
        return

    print("\n=== SETUP GARIS COUNTING ===")
    print("• Klik 2 titik untuk garis LANE KIRI (kuning)")
    print("• Klik 2 titik untuk garis LANE KANAN (pink)")
    print("• Tekan R untuk reset, ENTER untuk konfirmasi\n")

    selector = LineSelector(first_frame)
    lane_kiri, lane_kanan = selector.run()

    if lane_kiri is None or lane_kanan is None:
        print("Setup dibatalkan.")
        return

    print(f"Lane Kiri  : {lane_kiri}")
    print(f"Lane Kanan : {lane_kanan}")
    print("\nMemulai processing video...\n")

    processor = VideoProcessor(args.video, lane_kiri, lane_kanan)
    processor.run()


if __name__ == "__main__":
    main()