import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

conn = sqlite3.connect("antrenman_analiz.db")
cur = conn.cursor()

# Check row counts across 3NF tables
for tbl in ["videos", "frames", "keypoints", "angles"]:
    cur.execute(f"SELECT COUNT(*) FROM {tbl}")
    print(f"Table '{tbl}': {cur.fetchone()[0]} rows")

cur.execute("""
    SELECT video_id, filename, fps, predicted_exercise, confidence, rep_count, primary_joint, rom_deg, form_score, form_status
    FROM videos ORDER BY video_id
""")
videos = cur.fetchall()

fig, axes = plt.subplots(4, 2, figsize=(16, 15), dpi=150)
fig.patch.set_facecolor('#F8FAFC')
axes = axes.flatten()

palette = {
    "chest_fly_angle": "#2563EB",
    "elbow_angle": "#059669",
    "shoulder_angle": "#7C3AED"
}

for idx, v in enumerate(videos):
    vid_id, fname, fps, ex_name, conf, reps, p_joint, rom, f_score, f_status = v
    ax = axes[idx]
    ax.set_facecolor('#FFFFFF')
    
    cur.execute("""
        SELECT f.timestamp_sec, a.raw_angle_deg, a.smoothed_angle_deg, a.is_valid_form
        FROM angles a
        JOIN frames f ON a.frame_id = f.frame_id
        WHERE f.video_id = ? AND a.joint_name = ?
        ORDER BY f.frame_index
    """, (vid_id, p_joint))
    rows = cur.fetchall()
    ts = np.array([r[0] for r in rows])
    raw_a = np.array([r[1] for r in rows])
    sm_a = np.array([r[2] for r in rows])
    valid_a = np.array([r[3] for r in rows])
    
    min_dist = max(3, int(fps * 1.25))
    peaks, _ = find_peaks(sm_a, distance=min_dist, prominence=13.5)
    troughs, _ = find_peaks(-sm_a, distance=min_dist, prominence=13.5)
    
    color = palette.get(p_joint, "#1E40AF")
    ax.plot(ts, raw_a, color=color, alpha=0.25, linewidth=1.2, label="Ham Aci (atan2)")
    ax.plot(ts, sm_a, color=color, linewidth=2.4, label=f"Moving Avg ({p_joint})")
    
    if len(peaks) > 0:
        ax.scatter(ts[peaks], sm_a[peaks], color="#DC2626", s=55, zorder=5, label=f"Tepe Noktasi ({len(peaks)})")
    if len(troughs) > 0:
        ax.scatter(ts[troughs], sm_a[troughs], color="#D97706", marker="^", s=55, zorder=5, label=f"Cukur Noktasi ({len(troughs)})")
        
    # Highlight invalid form regions
    if np.any(valid_a == 0):
        bad_ts = ts[valid_a == 0]
        ax.axvspan(bad_ts[0], bad_ts[-1], color="#FEE2E2", alpha=0.75, label="Form Bozulmasi (Bel < 150°)")
        
    badge_color = "#059669" if f_score >= 90 else "#DC2626"
    ax.set_title(
        f"{fname} | {ex_name}\nTekrar: {reps} Rep | ROM: {rom:.1f}° | Form: {f_score:.1f}/100",
        fontsize=10.5, fontweight="bold", color="#0F172A", pad=8
    )
    ax.set_xlabel("Zaman (sn)", fontsize=9, color="#334155")
    ax.set_ylabel("Aci (Derece °)", fontsize=9, color="#334155")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

# 8th subplot: Summary Architecture & 3NF Database Stats Table
ax8 = axes[7]
ax8.axis("off")
summary_text = (
    "3NF SQLITE VERİTABANI VE MODEL ÖZETİ\n"
    "────────────────────────────────────────────────────\n"
    "• Mimari: OpenCV + MediaPipe/Kinematik + SciPy + SQLite (3NF)\n"
    "• İncelenen Klasör: Google Drive / 'Spor algılama'\n"
    "• İşlenen Video Sayısı: 7 MP4 Antrenman Videosu (1.869 Kare)\n"
    "• 3NF Tablo Kayıtları:\n"
    "   - videos    : 7 video özeti ve model sınıflandırma sonucu\n"
    "   - frames    : 1.869 zaman damgalı video karesi\n"
    "   - keypoints : 61.677 adet 3B eklem koordinatı (visibility > 0.75)\n"
    "   - angles    : 9.345 adet trigonometrik (atan2) eklem açısı\n"
    "• Sınıflandırılan 4 Egzersiz Grubu:\n"
    "   1. Pec Deck / Machine Chest Fly (2 Video - 6 Toplam Tekrar)\n"
    "   2. Incline Chest Press Machine (2 Video - 5 Toplam Tekrar)\n"
    "   3. Seated Dumbbell Shoulder Press (1 Video - 3 Toplam Tekrar)\n"
    "   4. Cable Triceps Pushdown (2 Video - 6 Toplam Tekrar)\n"
    "• Form Analizi Tespiti:\n"
    "   - 6 videoda (%85.7) yeşil iskelet / kusursuz anatomik form\n"
    "   - VID20261026140133.mp4 son %12.6'lık bölümde bel eğimi uyarısı"
)
ax8.text(
    0.03, 0.95, summary_text,
    transform=ax8.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.7", facecolor="#EFF6FF", edgecolor="#93C5FD", linewidth=1.5),
    family="monospace",
    color="#1E293B"
)

plt.suptitle(
    "Bilgisayarlı Görü ve Zaman Serisi Analizi ile Otonom Antrenman Form Analizi ve Tekrar Sayma Sistemi\n"
    "('Spor algılama' Klasöründeki Videoların Zaman Serisi Açı Sinyalleri ve scipy.signal.find_peaks Çıktıları)",
    fontsize=13.5, fontweight="bold", color="#0F172A", y=0.99
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_img = "spor_algilama_zaman_serisi_analizi.png"
plt.savefig(out_img, dpi=150)
print(f"Saved dashboard to {out_img}")
conn.close()
