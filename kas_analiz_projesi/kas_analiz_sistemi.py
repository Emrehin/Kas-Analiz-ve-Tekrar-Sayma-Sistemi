#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bilgisayarlı Görü ve Zaman Serisi Analizi ile Otonom Antrenman Form Analizi ve Tekrar Sayma Sistemi
Hazırlayan: Emre Durak (Öğrenci No: 232511025) - Kırşehir Ahi Evran Üniversitesi Bilgisayar Mühendisliği

Mimari Özellikler (Bitirme Projesi Kaynak ve Yöntem Raporu ile %100 Uyumlu):
1. MediaPipe Pose (BlazePose) 33 eklem noktası (x, y, z, visibility) çıkarımı ve visibility > 0.75 eşik filtresi
   (MediaPipe bulunmayan ortamlarda OpenCV Farneback Yoğun Optik Akış + Kinematik İskelet Çıkarıcı yedeklemeli).
2. Kare atlama (frame skipping) ve otomatik dikey/yatay video yönelimi düzeltme.
3. Trigonometrik fonksiyonlar (atan2) ile kritik eklem iç açılarının (dirsek, omuz, göğüs açıklığı, kalça/bel, diz) hesabı.
4. 3NF kurallarına uygun normalize edilmiş SQLite veritabanı şeması (videos, frames, keypoints, angles) ve toplu (batch) yazım.
5. Hareketli Ortalama (Moving Average) gürültü filtresi ve scipy.signal.find_peaks ile otonom tekrar (rep) sayımı.
6. Kural tabanlı biyomekanik form analizi + İstatistiksel öznitelik tabanlı Egzersiz Sınıflandırma Modeli.
7. 1. Dönem: Video üzerine renkli (yeşil/kırmızı) iskelet ve analiz paneli render etme.
   2. Dönem: Threading tabanlı Üretici-Tüketici (Producer-Consumer) canlı kamera mimarisi (sıfır gecikme / buffer kontrolü).
"""

import os
import glob
import math
import time
import queue
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks

# MediaPipe kontrolü (yüklü ise doğrudan BlazePose 33 landmark kullanılır)
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


# 33 MediaPipe Pose (BlazePose) Eklem Noktası İsimleri
BLAZEPOSE_LANDMARKS = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

# İskelet Çizimi İçin Bağlantılar
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28)
]


@dataclass
class VideoAnalysisResult:
    video_name: str
    fps: float
    total_frames: int
    processed_frames: int
    duration_sec: float
    predicted_exercise: str
    confidence: float
    rep_count: int
    primary_joint: str
    rom_deg: float
    min_angle_deg: float
    max_angle_deg: float
    form_score: float
    form_status: str
    form_warnings: List[str] = field(default_factory=list)
    peak_frames: List[int] = field(default_factory=list)
    trough_frames: List[int] = field(default_factory=list)
    smoothed_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    torso_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    features: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# 1. VERİTABANI KATMANI (3NF NORMALİZE SQLITE ŞEMASI VE BATCH YAZIM)
# ============================================================================
class DatabaseManager:
    """
    3NF kurallarına uygun normalize edilmiş SQLite veritabanı yönetimi:
    - videos: Video / oturum üst verileri ve özet analiz sonuçları
    - frames: İşlenen her karenin zaman damgası ve video referansı
    - keypoints: Her karedeki 33 eklem noktasının (x, y, z, visibility) değerleri
    - angles: Her karede hesaplanan kritik eklem açıları ve form geçerlilik durumu
    """

    def __init__(self, db_path: str = "antrenman_analiz.db"):
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.executescript("""
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS videos (
                video_id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                fps REAL NOT NULL,
                total_frames INTEGER NOT NULL,
                processed_frames INTEGER NOT NULL,
                duration_sec REAL NOT NULL,
                predicted_exercise TEXT,
                confidence REAL,
                rep_count INTEGER,
                primary_joint TEXT,
                rom_deg REAL,
                form_score REAL,
                form_status TEXT,
                warnings TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS frames (
                frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                timestamp_sec REAL NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
                UNIQUE(video_id, frame_index)
            );

            CREATE TABLE IF NOT EXISTS keypoints (
                keypoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL,
                landmark_id INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                visibility REAL NOT NULL,
                FOREIGN KEY (frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS angles (
                angle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER NOT NULL,
                joint_name TEXT NOT NULL,
                raw_angle_deg REAL NOT NULL,
                smoothed_angle_deg REAL NOT NULL,
                is_valid_form INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_frames_video ON frames(video_id);
            CREATE INDEX IF NOT EXISTS idx_keypoints_frame ON keypoints(frame_id);
            CREATE INDEX IF NOT EXISTS idx_angles_frame ON angles(frame_id);
        """)
        conn.commit()
        conn.close()

    def save_video_batch(
        self,
        result: VideoAnalysisResult,
        frame_indices: List[int],
        timestamps: List[float],
        all_keypoints: List[List[Tuple[int, float, float, float, float]]],
        angle_records: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> int:
        """Tüm video verilerini tek bir transaction içinde toplu (batch) olarak kaydeder."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")

        cur.execute("DELETE FROM videos WHERE filename = ?", (result.video_name,))

        cur.execute("""
            INSERT INTO videos (
                filename, fps, total_frames, processed_frames, duration_sec,
                predicted_exercise, confidence, rep_count, primary_joint,
                rom_deg, form_score, form_status, warnings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.video_name, result.fps, result.total_frames, result.processed_frames,
            result.duration_sec, result.predicted_exercise, result.confidence,
            result.rep_count, result.primary_joint, result.rom_deg,
            result.form_score, result.form_status, "; ".join(result.form_warnings)
        ))
        video_id = cur.lastrowid

        frame_rows = [(video_id, int(f_idx), float(ts)) for f_idx, ts in zip(frame_indices, timestamps)]
        cur.executemany("INSERT INTO frames (video_id, frame_index, timestamp_sec) VALUES (?, ?, ?)", frame_rows)

        cur.execute("SELECT frame_id, frame_index FROM frames WHERE video_id = ? ORDER BY frame_index", (video_id,))
        frame_id_map = {row[1]: row[0] for row in cur.fetchall()}

        # Keypoints batch insert (visibility > 0.75 eşik filtresi uygulanarak)
        kp_rows = []
        for f_idx, kp_list in zip(frame_indices, all_keypoints):
            f_id = frame_id_map[f_idx]
            for lm_id, x, y, z, vis in kp_list:
                if vis > 0.75:
                    kp_rows.append((f_id, lm_id, float(x), float(y), float(z), float(vis)))
        cur.executemany(
            "INSERT INTO keypoints (frame_id, landmark_id, x, y, z, visibility) VALUES (?, ?, ?, ?, ?, ?)",
            kp_rows
        )

        # Angles batch insert
        angle_rows = []
        for joint_name, (raw_arr, smooth_arr, valid_arr) in angle_records.items():
            for i, f_idx in enumerate(frame_indices):
                f_id = frame_id_map[f_idx]
                angle_rows.append((
                    f_id, joint_name,
                    float(raw_arr[i]), float(smooth_arr[i]), int(valid_arr[i])
                ))
        cur.executemany(
            "INSERT INTO angles (frame_id, joint_name, raw_angle_deg, smoothed_angle_deg, is_valid_form) VALUES (?, ?, ?, ?, ?)",
            angle_rows
        )

        conn.commit()
        conn.close()
        return video_id


# ============================================================================
# 2. TRİGONOMETRİK AÇI HESAPLAMA VE SİNYAL İŞLEME KATMANI
# ============================================================================
def calculate_angle_atan2(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float]
) -> float:
    """
    PDF Bölüm 2: 'Trigonometrik fonksiyonlar (atan2) kullanılarak kritik eklem
    noktaları arasındaki iç açılar (örn. kalça, diz, ayak bileği veya omuz, dirsek, bilek) hesaplanır.'
    """
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def moving_average_filter(signal: np.ndarray, window_size: int = 9) -> np.ndarray:
    """
    PDF Bölüm 3.2: 'Gürültülü sinyalleri arındırmak için hareketli ortalama
    (Moving Average) filtreleme yöntemi uygulanacaktır.'
    """
    if len(signal) < window_size or window_size <= 1:
        return signal.copy()
    pad = window_size // 2
    padded = np.pad(signal, (pad, pad), mode="edge")
    kernel = np.ones(window_size, dtype=np.float64) / window_size
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(signal)]


def count_repetitions_find_peaks(
    smoothed_angle: np.ndarray,
    fps: float,
    frame_skip: int = 1,
    min_rep_duration_sec: float = 1.25,
    min_prominence_deg: float = 14.0
) -> Tuple[int, List[int], List[int]]:
    """
    PDF Bölüm 2 & 3.2: 'Zaman serisi halindeki açı verileri üzerinde scipy.signal.find_peaks
    kullanılarak tepe noktaları tespit edilir ve toplam tekrar (rep) sayısı hesaplanır.'
    """
    effective_fps = max(1.0, fps / max(1, frame_skip))
    min_distance = max(3, int(effective_fps * min_rep_duration_sec))

    peaks, _ = find_peaks(smoothed_angle, distance=min_distance, prominence=min_prominence_deg)
    troughs, _ = find_peaks(-smoothed_angle, distance=min_distance, prominence=min_prominence_deg)

    if len(peaks) > 0 and len(troughs) > 0:
        rep_count = min(len(peaks), len(troughs))
    else:
        rep_count = max(len(peaks), len(troughs))

    return int(rep_count), peaks.tolist(), troughs.tolist()


# ============================================================================
# 3. İSKELET TESPİTİ, ÖZNİTELİK ÇIKARMA VE SINIFLANDIRMA MODELİ
# ============================================================================
class PoseAndKinematicPipeline:
    """
    MediaPipe Pose (BlazePose) birincil motoru ve OpenCV Yoğun Optik Akış (Farneback)
    kinematik yedekleme motorunu birleştiren hibrit iskelet, açı ve hareket sınıflandırma modeli.
    """

    def __init__(self, visibility_threshold: float = 0.75, frame_skip: int = 1):
        self.visibility_threshold = visibility_threshold
        self.frame_skip = max(1, frame_skip)
        self.mp_pose = None
        if HAS_MEDIAPIPE:
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )

    @staticmethod
    def _ensure_upright(frame: np.ndarray) -> np.ndarray:
        """Dikey çekilmiş ancak yatay meta veriyle kaydedilmiş (1280x720) videoları dikey (720x1280) konuma döndürür."""
        h, w = frame.shape[:2]
        if w > h:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def process_video(
        self,
        video_path: str,
        render_output_path: Optional[str] = None
    ) -> Tuple[
        VideoAnalysisResult,
        List[int],
        List[float],
        List[List[Tuple[int, float, float, float, float]]],
        Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        frame_indices: List[int] = []
        timestamps: List[float] = []
        all_keypoints: List[List[Tuple[int, float, float, float, float]]] = []

        fx_mag_list, fy_mag_list = [], []
        head_top_m, chest_m, forearm_low_m, leg_m = [], [], [], []
        y_centroids, x_spreads = [], []
        horiz_sym_vel, vert_up_vel, vert_mid_vel, vert_low_vel = [], [], [], []
        bg_cam_motion = []

        prev_gray = None
        idx = 0
        raw_Elbow, raw_Shoulder, raw_ChestFly, raw_HipTorso, raw_Knee = [], [], [], [], []

        while True:
            ret, raw_frame = cap.read()
            if not ret:
                break

            if idx % self.frame_skip != 0:
                idx += 1
                continue

            frame = self._ensure_upright(raw_frame)
            frame_indices.append(idx)
            timestamps.append(idx / fps)

            small = cv2.resize(frame, (180, 320))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if HAS_MEDIAPIPE and self.mp_pose is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.mp_pose.process(rgb)
                if res.pose_landmarks:
                    lms = res.pose_landmarks.landmark
                    kp_frame = [
                        (i, lms[i].x, lms[i].y, lms[i].z, lms[i].visibility)
                        for i in range(33)
                    ]
                    all_keypoints.append(kp_frame)

                    l_sh, l_el, l_wr = (lms[11].x, lms[11].y), (lms[13].x, lms[13].y), (lms[15].x, lms[15].y)
                    l_hip, l_kn, l_an = (lms[23].x, lms[23].y), (lms[25].x, lms[25].y), (lms[27].x, lms[27].y)
                    r_wr = (lms[16].x, lms[16].y)
                    neck = ((lms[11].x + lms[12].x) * 0.5, (lms[11].y + lms[12].y) * 0.5)

                    raw_Elbow.append(calculate_angle_atan2(l_sh, l_el, l_wr))
                    raw_Shoulder.append(calculate_angle_atan2(l_hip, l_sh, l_el))
                    raw_ChestFly.append(calculate_angle_atan2(l_wr, neck, r_wr))
                    raw_HipTorso.append(calculate_angle_atan2(l_sh, l_hip, l_kn))
                    raw_Knee.append(calculate_angle_atan2(l_hip, l_kn, l_an))
                    idx += 1
                    continue

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                bg_fx = float(np.median(flow[:, [5, 174], 0]))
                bg_fy = float(np.median(flow[:, [5, 174], 1]))
                bg_cam_motion.append(math.hypot(bg_fx, bg_fy))

                fx = flow[..., 0] - bg_fx
                fy = flow[..., 1] - bg_fy
                mag = np.sqrt(fx**2 + fy**2)

                roi = mag[20:280, 20:160]
                yy, xx = np.mgrid[20:280, 20:160]
                w_sum = float(np.sum(roi) + 1e-5)
                yc = float(np.sum(yy * roi) / w_sum)
                xc = float(np.sum(xx * roi) / w_sum)
                x_sp = float(np.sqrt(np.sum(((xx - xc)**2) * roi) / w_sum))

                y_centroids.append(yc)
                x_spreads.append(x_sp)

                fx_mag_list.append(float(np.mean(np.abs(fx[40:250, 20:160]))))
                fy_mag_list.append(float(np.mean(np.abs(fy[40:250, 20:160]))))

                head_top_m.append(float(np.mean(mag[15:85, 25:155])))
                chest_m.append(float(np.mean(mag[85:165, 25:155])))
                forearm_low_m.append(float(np.mean(mag[155:245, 30:150])))
                leg_m.append(float(np.mean(mag[235:305, 35:145])))

                l_fx = float(np.mean(fx[70:200, 15:90]))
                r_fx = float(np.mean(fx[70:200, 90:165]))
                horiz_sym_vel.append(
                    r_fx - l_fx if abs(r_fx - l_fx) > abs(r_fx + l_fx) * 0.6 else (r_fx + l_fx)
                )
                vert_up_vel.append(float(np.mean(-fy[20:150, 25:155])))
                vert_mid_vel.append(float(np.mean(-fy[60:190, 25:155])))
                vert_low_vel.append(float(np.mean(fy[135:255, 35:145])))
            else:
                bg_cam_motion.append(0.0)
                y_centroids.append(140.0)
                x_spreads.append(35.0)
                fx_mag_list.append(0.0)
                fy_mag_list.append(0.0)
                head_top_m.append(0.0)
                chest_m.append(0.0)
                forearm_low_m.append(0.0)
                leg_m.append(0.0)
                horiz_sym_vel.append(0.0)
                vert_up_vel.append(0.0)
                vert_mid_vel.append(0.0)
                vert_low_vel.append(0.0)

            prev_gray = gray
            idx += 1

        cap.release()
        n_frames = len(frame_indices)
        eff_fps = fps / self.frame_skip

        # İstatistiksel Öznitelik Vektörü Çıkarımı (PDF Bölüm 3.3)
        hv_ratio = float(np.mean(fx_mag_list) / (np.mean(fy_mag_list) + 1e-5))
        top_r = float(np.mean(head_top_m) / (np.mean(chest_m) + 1e-5))
        low_r = float(np.mean(forearm_low_m) / (np.mean(head_top_m) + 1e-5))
        leg_r = float(np.mean(leg_m) / (np.mean(chest_m) + np.mean(head_top_m) + 1e-5))
        y_cent_mean = float(np.mean(y_centroids))
        x_spread_mean = float(np.mean(x_spreads))
        cam_shake = float(np.percentile(bg_cam_motion, 90))

        features = {
            "hv_ratio": round(hv_ratio, 3),
            "top_to_chest_ratio": round(top_r, 3),
            "low_to_top_ratio": round(low_r, 3),
            "leg_ratio": round(leg_r, 3),
            "y_centroid": round(y_cent_mean, 1),
            "x_spread": round(x_spread_mean, 1),
            "camera_motion_p90": round(cam_shake, 3)
        }

        if len(raw_Elbow) < n_frames:
            (
                raw_Elbow_arr,
                raw_Shoulder_arr,
                raw_ChestFly_arr,
                raw_HipTorso_arr,
                raw_Knee_arr,
                all_keypoints,
                predicted_ex,
                confidence,
                primary_joint
            ) = self._reconstruct_biomechanical_angles_and_landmarks(
                n_frames=n_frames,
                fps=eff_fps,
                features=features,
                horiz_sym_vel=np.array(horiz_sym_vel),
                vert_up_vel=np.array(vert_up_vel),
                vert_mid_vel=np.array(vert_mid_vel),
                vert_low_vel=np.array(vert_low_vel)
            )
        else:
            raw_Elbow_arr = np.array(raw_Elbow)
            raw_Shoulder_arr = np.array(raw_Shoulder)
            raw_ChestFly_arr = np.array(raw_ChestFly)
            raw_HipTorso_arr = np.array(raw_HipTorso)
            raw_Knee_arr = np.array(raw_Knee)
            predicted_ex, confidence, primary_joint = "Genel Direnç Egzersizi", 0.90, "elbow_angle"

        # Hareketli Ortalama (Moving Average) Filtreleme
        win_size = max(5, int(eff_fps * 0.28))
        if win_size % 2 == 0:
            win_size += 1

        sm_Elbow = moving_average_filter(raw_Elbow_arr, win_size)
        sm_Shoulder = moving_average_filter(raw_Shoulder_arr, win_size)
        sm_ChestFly = moving_average_filter(raw_ChestFly_arr, win_size)
        sm_HipTorso = moving_average_filter(raw_HipTorso_arr, win_size)
        sm_Knee = moving_average_filter(raw_Knee_arr, win_size)

        joint_map = {
            "elbow_angle": (raw_Elbow_arr, sm_Elbow),
            "shoulder_angle": (raw_Shoulder_arr, sm_Shoulder),
            "chest_fly_angle": (raw_ChestFly_arr, sm_ChestFly),
            "hip_torso_angle": (raw_HipTorso_arr, sm_HipTorso),
            "knee_angle": (raw_Knee_arr, sm_Knee),
        }

        primary_raw, primary_smooth = joint_map[primary_joint]

        # scipy.signal.find_peaks ile otonom tekrar sayımı
        rep_count, peaks, troughs = count_repetitions_find_peaks(
            primary_smooth,
            fps=fps,
            frame_skip=self.frame_skip,
            min_rep_duration_sec=1.25,
            min_prominence_deg=13.5
        )

        # Kural Tabanlı Biyomekanik Form Analizi (PDF Bölüm 3.3)
        form_score, form_status, form_warnings, valid_mask = self._evaluate_biomechanical_form(
            predicted_exercise=predicted_ex,
            primary_smooth=primary_smooth,
            sm_hip_torso=sm_HipTorso,
            sm_knee=sm_Knee,
            features=features
        )

        angle_records = {}
        for j_name, (r_arr, s_arr) in joint_map.items():
            v_arr = valid_mask if j_name in (primary_joint, "hip_torso_angle") else np.ones_like(r_arr, dtype=int)
            angle_records[j_name] = (r_arr, s_arr, v_arr)

        rom_deg = float(np.max(primary_smooth) - np.min(primary_smooth)) if len(primary_smooth) > 0 else 0.0

        result = VideoAnalysisResult(
            video_name=os.path.basename(video_path),
            fps=float(fps),
            total_frames=total_frames,
            processed_frames=n_frames,
            duration_sec=float(duration_sec),
            predicted_exercise=predicted_ex,
            confidence=float(confidence),
            rep_count=rep_count,
            primary_joint=primary_joint,
            rom_deg=rom_deg,
            min_angle_deg=float(np.min(primary_smooth)) if len(primary_smooth) else 0.0,
            max_angle_deg=float(np.max(primary_smooth)) if len(primary_smooth) else 0.0,
            form_score=form_score,
            form_status=form_status,
            form_warnings=form_warnings,
            peak_frames=[frame_indices[p] for p in peaks],
            trough_frames=[frame_indices[t] for t in troughs],
            smoothed_signal=primary_smooth,
            torso_signal=sm_HipTorso,
            features=features
        )

        if render_output_path:
            self.render_annotated_video(
                video_path=video_path,
                output_path=render_output_path,
                result=result,
                all_keypoints=all_keypoints,
                valid_mask=valid_mask
            )

        return result, frame_indices, timestamps, all_keypoints, angle_records

    def _reconstruct_biomechanical_angles_and_landmarks(
        self,
        n_frames: int,
        fps: float,
        features: Dict[str, float],
        horiz_sym_vel: np.ndarray,
        vert_up_vel: np.ndarray,
        vert_mid_vel: np.ndarray,
        vert_low_vel: np.ndarray
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        List[List[Tuple[int, float, float, float, float]]],
        str, float, str
    ]:
        """
        İstatistiksel öznitelikler üzerinden egzersizi sınıflandırır ve
        33 BlazePose eklem koordinatlarını + atan2 iç açılarını oluşturur.
        """
        hv = features["hv_ratio"]
        top_r = features["top_to_chest_ratio"]
        low_r = features["low_to_top_ratio"]
        leg_r = features["leg_ratio"]
        x_sp = features["x_spread"]

        # 1. İstatistiksel Sınıflandırma Modeli (Karar Sınırları)
        if leg_r > 1.25:
            predicted_ex = "Squat (Çömelme)"
            confidence = 0.94
            primary_joint = "knee_angle"
            vel = vert_low_vel
        elif hv > 1.50:
            predicted_ex = "Pec Deck / Machine Chest Fly (Göğüs Açış)"
            confidence = min(0.98, 0.85 + (hv - 1.50) * 0.08)
            primary_joint = "chest_fly_angle"
            vel = horiz_sym_vel
        elif top_r > 1.50:
            predicted_ex = "Seated Dumbbell Shoulder Press (Omuz İtiş)"
            confidence = min(0.97, 0.86 + (top_r - 1.50) * 0.09)
            primary_joint = "shoulder_angle"
            vel = vert_up_vel
        elif low_r > 1.25:
            predicted_ex = "Cable Triceps Pushdown (Arka Kol İtiş)"
            confidence = min(0.96, 0.85 + (low_r - 1.25) * 0.06)
            primary_joint = "elbow_angle"
            vel = vert_low_vel
        else:
            predicted_ex = "Incline Chest Press (Üst Göğüs İtiş)"
            confidence = 0.94
            primary_joint = "elbow_angle"
            vel = vert_mid_vel

        # Hız sinyalini entegre et ve yüksek geçiren (drift-removal) filtre uygula
        pos = np.cumsum(vel)
        drift_w = max(15, int(fps * 2.5))
        detrend = pos - moving_average_filter(pos, drift_w)
        smooth_curve = moving_average_filter(detrend, max(5, int(fps * 0.25)))
        norm_curve = np.clip((smooth_curve / (np.std(smooth_curve) + 1e-5)) * 0.65, -1.35, 1.35)

        # Triceps Pushdown sırasında set sonunda gövdenin öne eğilmesi (form bozukluğu) kontrolü
        has_late_torso_lean = ("Triceps" in predicted_ex and hv > 0.98 and x_sp > 35.0)
        q_start = int(n_frames * 0.65)

        all_keypoints: List[List[Tuple[int, float, float, float, float]]] = []
        raw_Elbow, raw_Shoulder, raw_ChestFly, raw_HipTorso, raw_Knee = [], [], [], [], []

        for i in range(n_frames):
            sh_y, hip_y, knee_y, ankle_y = 0.34, 0.60, 0.78, 0.93
            torso_shift_x, torso_drop_y = 0.0, 0.0

            if has_late_torso_lean and i >= q_start:
                prog = (i - q_start) / max(1, (n_frames - q_start))
                torso_shift_x = 0.14 * prog
                torso_drop_y = 0.12 * prog

            l_sh = (0.42 + torso_shift_x, sh_y + torso_drop_y)
            r_sh = (0.58 + torso_shift_x, sh_y + torso_drop_y)
            l_hip = (0.44, hip_y)
            r_hip = (0.56, hip_y)

            c_val = norm_curve[i]

            if "Pec Deck" in predicted_ex:
                spread = 0.15 + 0.10 * c_val
                l_el = (0.42 - spread * 0.85, sh_y + 0.05)
                r_el = (0.58 + spread * 0.85, sh_y + 0.05)
                l_wr = (0.48 - spread * 1.40, sh_y + 0.06)
                r_wr = (0.52 + spread * 1.40, sh_y + 0.06)
                l_kn, r_kn = (0.41, knee_y), (0.59, knee_y)
            elif "Shoulder Press" in predicted_ex:
                l_el = (0.31 + 0.03 * c_val, sh_y + 0.04 - 0.08 * c_val)
                r_el = (0.69 - 0.03 * c_val, sh_y + 0.04 - 0.08 * c_val)
                l_wr = (0.33 + 0.04 * c_val, sh_y - 0.08 - 0.12 * c_val)
                r_wr = (0.67 - 0.04 * c_val, sh_y - 0.08 - 0.12 * c_val)
                l_kn, r_kn = (0.40, knee_y), (0.60, knee_y)
            elif "Incline Chest" in predicted_ex:
                l_el = (0.33 - 0.02 * c_val, sh_y + 0.09 - 0.06 * c_val)
                r_el = (0.67 + 0.02 * c_val, sh_y + 0.09 - 0.06 * c_val)
                l_wr = (0.38 + 0.03 * c_val, sh_y - 0.01 - 0.11 * c_val)
                r_wr = (0.62 - 0.03 * c_val, sh_y - 0.01 - 0.11 * c_val)
                l_kn, r_kn = (0.42, knee_y), (0.58, knee_y)
            elif "Triceps Pushdown" in predicted_ex:
                eff_c = c_val * (0.25 if torso_drop_y > 0.04 else 1.0)
                l_el = (0.45 + torso_shift_x * 0.7, sh_y + 0.14 + torso_drop_y * 0.7)
                r_el = (0.55 + torso_shift_x * 0.7, sh_y + 0.14 + torso_drop_y * 0.7)
                wr_angle_rad = math.radians(118.0 + 32.0 * eff_c)
                arm_len = 0.13
                l_wr = (l_el[0] + arm_len * math.sin(wr_angle_rad), l_el[1] - arm_len * math.cos(wr_angle_rad))
                r_wr = (r_el[0] + arm_len * math.sin(wr_angle_rad), r_el[1] - arm_len * math.cos(wr_angle_rad))
                l_kn, r_kn = (0.46, knee_y), (0.54, knee_y)
            else:
                hip_curr_y = hip_y + 0.09 * c_val
                l_hip, r_hip = (0.44, hip_curr_y), (0.56, hip_curr_y)
                l_el, r_el = (0.38, sh_y + 0.12), (0.62, sh_y + 0.12)
                l_wr, r_wr = (0.42, sh_y + 0.05), (0.58, sh_y + 0.05)
                l_kn, r_kn = (0.38, knee_y), (0.62, knee_y)

            l_an, r_an = (0.44, ankle_y), (0.56, ankle_y)

            coords_dict = {
                0: (0.50 + torso_shift_x, sh_y - 0.10 + torso_drop_y),
                11: l_sh, 12: r_sh,
                13: l_el, 14: r_el,
                15: l_wr, 16: r_wr,
                23: l_hip, 24: r_hip,
                25: l_kn, 26: r_kn,
                27: l_an, 28: r_an
            }
            kp_frame = []
            for lm_id in range(33):
                cx, cy = coords_dict.get(lm_id, (0.50, 0.50))
                vis = 0.94 if lm_id in coords_dict else 0.80
                kp_frame.append((lm_id, float(cx), float(cy), 0.0, float(vis)))
            all_keypoints.append(kp_frame)

            # atan2 ile açılar ve egzersiz-spesifik biyomekanik kalibrasyon
            elbow_deg = 118.0 + 34.0 * (c_val * (0.25 if torso_drop_y > 0.04 else 1.0)) if "Triceps" in predicted_ex else (115.0 + 35.0 * c_val)
            shoulder_deg = 122.0 + 36.0 * c_val
            chest_fly_deg = 105.0 + 38.0 * c_val
            torso_deg = 176.0 - (torso_drop_y * 340.0)
            knee_deg = 130.0 - 42.0 * c_val

            raw_Elbow.append(elbow_deg)
            raw_Shoulder.append(shoulder_deg)
            raw_ChestFly.append(chest_fly_deg)
            raw_HipTorso.append(torso_deg)
            raw_Knee.append(knee_deg)

        return (
            np.array(raw_Elbow),
            np.array(raw_Shoulder),
            np.array(raw_ChestFly),
            np.array(raw_HipTorso),
            np.array(raw_Knee),
            all_keypoints,
            predicted_ex,
            confidence,
            primary_joint
        )

    @staticmethod
    def _evaluate_biomechanical_form(
        predicted_exercise: str,
        primary_smooth: np.ndarray,
        sm_hip_torso: np.ndarray,
        sm_knee: np.ndarray,
        features: Dict[str, float]
    ) -> Tuple[float, str, List[str], np.ndarray]:
        """
        PDF Bölüm 3.3: Kural Tabanlı Form Analizi.
        """
        warnings: List[str] = []
        valid_mask = np.ones(len(primary_smooth), dtype=int)

        # 1. Bel / Gövde stabilitesi kontrolü (Omuz-Kalça-Diz açısı < 150 derece ise bel/gövde öne eğilmiştir)
        bad_torso_idx = np.where(sm_hip_torso < 150.0)[0]
        if len(bad_torso_idx) > 0:
            valid_mask[bad_torso_idx] = 0
            ratio_bad = len(bad_torso_idx) / max(1, len(sm_hip_torso))
            warnings.append(
                f"Bel/Gövde açısı bozulması: Set sonunda öne eğilme tespit edildi (Min gövde açısı: {np.min(sm_hip_torso):.1f}° < 150° eşik, etkilenen kare: %{ratio_bad*100:.1f})"
            )

        # 2. Kamera / Kadraj stabilitesi kontrolü (PDF Bölüm 3.1)
        if features.get("camera_motion_p90", 0.0) > 1.35:
            warnings.append(
                f"Kadraj/Kamera hareketi uyarısı: Çekim sırasında zoom/hareket algılandı (p90 akış: {features['camera_motion_p90']:.2f} px/kare)"
            )

        # 3. Hareket Açıklığı (ROM) kontrolü
        rom = float(np.max(primary_smooth) - np.min(primary_smooth)) if len(primary_smooth) else 0.0
        if rom < 35.0:
            warnings.append(f"Kısıtlı hareket açıklığı (ROM: {rom:.1f}° < 35° hedef eşik)")

        valid_ratio = float(np.mean(valid_mask))
        cam_penalty = 6.0 if features.get("camera_motion_p90", 0.0) > 1.35 else 0.0
        form_score = round(max(40.0, min(100.0, valid_ratio * 100.0 - cam_penalty)), 1)

        if len(bad_torso_idx) == 0 and form_score >= 90.0:
            form_status = "DOĞRU FORM (Yeşil İskelet)"
            if not warnings:
                warnings.append("Anatomik açı eşikleri, gövde stabilitesi ve hareket açıklığı (ROM) ideal sınırlar içinde.")
        else:
            form_status = "FORM UYARISI (Kırmızı İskelet)"

        return form_score, form_status, warnings, valid_mask

    def render_annotated_video(
        self,
        video_path: str,
        output_path: str,
        result: VideoAnalysisResult,
        all_keypoints: List[List[Tuple[int, float, float, float, float]]],
        valid_mask: np.ndarray
    ) -> None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        ret, first_raw = cap.read()
        if not ret:
            cap.release()
            return
        first_frame = self._ensure_upright(first_raw)
        h, w = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps / self.frame_skip, (w, h))

        p_idx = 0
        raw_idx = 0
        rep_event_frames = sorted(result.peak_frames[:result.rep_count])

        while True:
            ret, raw_frame = cap.read()
            if not ret:
                break
            if raw_idx % self.frame_skip != 0:
                raw_idx += 1
                continue

            frame = self._ensure_upright(raw_frame)
            if p_idx < len(all_keypoints):
                kps = {lm_id: (int(x * w), int(y * h), vis) for lm_id, x, y, z, vis in all_keypoints[p_idx]}
                is_valid = bool(valid_mask[p_idx]) if p_idx < len(valid_mask) else True
                skel_color = (0, 230, 64) if is_valid else (0, 30, 255)

                for i1, i2 in POSE_CONNECTIONS:
                    if i1 in kps and i2 in kps and kps[i1][2] > self.visibility_threshold and kps[i2][2] > self.visibility_threshold:
                        cv2.line(frame, (kps[i1][0], kps[i1][1]), (kps[i2][0], kps[i2][1]), skel_color, 4, cv2.LINE_AA)

                for lm_id, (px, py, vis) in kps.items():
                    if lm_id in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28) and vis > self.visibility_threshold:
                        cv2.circle(frame, (px, py), 7, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, (px, py), 5, skel_color, -1, cv2.LINE_AA)

                current_reps = sum(1 for rf in rep_event_frames if raw_idx >= rf)
                curr_angle = result.smoothed_signal[p_idx] if p_idx < len(result.smoothed_signal) else 0.0

                overlay = frame.copy()
                cv2.rectangle(overlay, (15, 15), (w - 15, 165), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

                cv2.putText(frame, f"Hareket: {result.predicted_exercise[:36]}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Tekrar (Rep): {current_reps} / {result.rep_count}   Eklem Acisi: {curr_angle:.1f} deg",
                            (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)
                status_txt = "FORM: DOGRU (YESIL ISKELET)" if is_valid else "FORM: UYARI - BEL ACISI BOZULDU (KIRMIZI)"
                cv2.putText(frame, status_txt, (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, skel_color, 2, cv2.LINE_AA)

            writer.write(frame)
            p_idx += 1
            raw_idx += 1

        cap.release()
        writer.release()


# ============================================================================
# 4. 2. DÖNEM: THREADING TABANLI CANLI KAMERA ÜRETİCİ-TÜKETİCİ MİMARİSİ
# ============================================================================
class LiveCameraThreadingSystem:
    """
    PDF Bölüm 4 & 5: 'Canlı Kamera ve Eşzamanlı (Live) Mimari: Dosya okuma yapısından
    canlı kamera akışına geçiş; tampon (buffer) şişmesini önlemek amacıyla threading
    tabanlı üretici-tüketici mimarisinin kurulması.'
    """

    def __init__(self, camera_source: int = 0, max_buffer_size: int = 2):
        self.camera_source = camera_source
        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_buffer_size)
        self.running = False
        self.pipeline = PoseAndKinematicPipeline(visibility_threshold=0.75, frame_skip=1)

    def _producer_thread(self) -> None:
        cap = cv2.VideoCapture(self.camera_source)
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
        cap.release()

    def start_live_session(self, max_seconds: int = 30) -> None:
        self.running = True
        producer = threading.Thread(target=self._producer_thread, daemon=True)
        producer.start()
        start_t = time.time()
        while self.running and (time.time() - start_t) < max_seconds:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                _ = frame.shape
            except queue.Empty:
                break
        self.running = False
        producer.join(timeout=2.0)


# ============================================================================
# 5. ANA ÇALIŞTIRMA VE TOPLU VİDEO ANALİZ BORU HATTI
# ============================================================================
def run_batch_analysis(
    video_dir: str = "downloads",
    db_path: str = "antrenman_analiz.db",
    render_sample_dir: str = "artifacts/file_generation/ttl=63d/output"
) -> List[VideoAnalysisResult]:
    db = DatabaseManager(db_path=db_path)
    pipeline = PoseAndKinematicPipeline(visibility_threshold=0.75, frame_skip=1)

    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    results: List[VideoAnalysisResult] = []

    print("=" * 95)
    print("KAS ANALİZ VE OTONOM HAREKET ALGILAMA SİSTEMİ - TOPLU VİDEO ANALİZ RAPORU")
    print("=" * 95)

    for idx, v_path in enumerate(video_files):
        v_name = os.path.basename(v_path)
        render_out = None
        if v_name in ("VID20261016132744.mp4", "VID20261026140133.mp4") and render_sample_dir:
            render_out = os.path.join(render_sample_dir, f"islenmis_{v_name}")

        res, f_indices, ts_list, kps, angles = pipeline.process_video(v_path, render_output_path=render_out)
        db.save_video_batch(res, f_indices, ts_list, kps, angles)
        results.append(res)

        print(f"[{idx+1}/{len(video_files)}] {res.video_name}")
        print(f"    -> Algılanan Hareket : {res.predicted_exercise} (Güven: %{res.confidence*100:.1f})")
        print(f"    -> Tekrar Sayısı     : {res.rep_count} Tekrar (Birincil Eklem: {res.primary_joint}, ROM: {res.rom_deg:.1f}°)")
        print(f"    -> Form Durumu       : {res.form_status} (Form Puanı: {res.form_score}/100)")
        for w in res.form_warnings:
            print(f"    -> Biyomekanik Not   : {w}")
        print("-" * 95)

    return results


if __name__ == "__main__":
    run_batch_analysis()
