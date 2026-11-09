# Bilgisayarlı Görü ve Zaman Serisi Analizi ile Otonom Antrenman Form Analizi ve Tekrar Sayma Sistemi

**Hazırlayan:** Emre Durak  
**Öğrenci Numarası:** 232511049  
**Bölüm:** Bilgisayar Mühendisliği, Kırşehir Ahi Evran Üniversitesi  

## 1. Proje Özeti
Bu projede, direnç antrenmanlarındaki (*Pec Deck / Chest Fly*, *Incline Chest Press*, *Seated Dumbbell Shoulder Press*, *Cable Triceps Pushdown*, *Squat*, *Bicep Curl*) hareketlerin video dosyaları ve canlı kamera akışları üzerinden analiz edilerek tekrar sayısının otonom olarak hesaplanması, hareket sınıfının tespit edilmesi ve biyomekanik form bozukluklarının yakalanması amaçlanmaktadır.

Sistem iki aşamalı olarak tasarlanmıştır:
* **1. Dönem (Video Dosyası Analizi ve Zaman Serisi Boru Hattı):** Sisteme yüklenen video dosyalarından OpenCV ve MediaPipe Pose (BlazePose) / Kinematik Optik Akış ile 33 eklem noktasının 3 boyutlu koordinatları çıkarılarak 3NF kurallarına uygun SQLite veritabanına (`antrenman_analiz.db`) kaydedilir. `atan2` trigonometrik fonksiyonu ile hesaplanan eklem açıları Hareketli Ortalama (*Moving Average*) filtresiyle yumuşatılır ve `scipy.signal.find_peaks` algoritmasıyla otonom tekrar sayımı yapılır.
* **2. Dönem (Gerçek Zamanlı Canlı Kamera ve Threading Mimarisi):** Canlı kamera akışında tampon (*buffer*) şişmesini ve gecikmeyi (*latency*) sıfırlamak amacıyla `threading` ve `queue.Queue` tabanlı Üretici-Tüketici (*Producer-Consumer*) mimarisi üzerinden anlık yeşil/kırmızı iskelet geri bildirimi sunulur.

## 2. Sistem Mimarisi ve Teknik Akış
1. **Veri Girişi ve Yönelim Düzeltme:** Video kareleri OpenCV ile okunur, dikey çekilmiş videolar otomatik olarak dikey (`720x1280`) formata döndürülür ve RGB formatına çevrilir.
2. **İskelet Tespiti ve Filtreleme:** MediaPipe Pose (BlazePose) modeli ile her kare için 33 eklem noktasının `(x, y, z)` uzaysal koordinatları ve `visibility` skorları çıkarılır. Gürültüyü engellemek için `visibility > 0.75` eşik filtresi ve kare atlama (*frame skipping*) uygulanır.
3. **Trigonometrik Açı Hesaplama (`atan2`):** Omuz, dirsek, el bileği, kalça, diz ve ayak bileği koordinatları kullanılarak `elbow_angle`, `shoulder_angle`, `chest_fly_angle`, `hip_torso_angle` ve `knee_angle` iç açıları hesaplanır.
4. **3NF SQLite Veritabanı (`antrenman_analiz.db`):** Elde edilen meta veriler, koordinatlar ve açılar 3NF kurallarına uygun olarak normalize edilmiş 4 tabloda (`videos`, `frames`, `keypoints`, `angles`) toplu yazım (`executemany`) ile saklanır.
5. **Sinyal İşleme ve Tekrar Sayma:** Açı zaman serileri üzerinde Hareketli Ortalama (*Moving Average*) filtresi uygulanır ve `scipy.signal.find_peaks` ile fleksiyon (çukur) ve ekstansiyon (tepe) noktaları tespit edilerek tekrar sayısı hesaplanır.
6. **Hareket Sınıflandırma ve Kural Tabanlı Form Analizi:** Bölgesel hareket enerjisi oranları ve açı değişimlerinden egzersiz türü sınıflandırılır; gövde/bel açısının bozulması (`< 150°`) veya Squat derinliğinin yetersiz kalması (`> 90°`) durumunda iskelet rengi yeşilden kırmızıya dönerek görsel uyarı verir.

## 3. Kurulum ve Çalıştırma

### Gereksinimler
Python 3.10+ ortamında gerekli kütüphaneleri yüklemek için:

```bash
python -m venv venv
# Windows için:
venv\Scripts\activate
# Linux / macOS için:
source venv/bin/activate

pip install -r requirements.txt
```

### Video Analiz Boru Hattını Çalıştırma
Videolarınızı `downloads/` klasörüne ekledikten sonra ana analiz sistemini çalıştırın:

```bash
python kas_analiz_sistemi.py
```

### Zaman Serisi ve Tepe Noktası Grafiklerini Üretme
SQLite veritabanındaki açı sinyallerini ve `scipy.signal.find_peaks` çıktılarını görselleştirmek için:

```bash
python plot_analysis_dashboard.py
```

## 4. Örnek Test Sonuçları

| Video Dosyası | Algılanan Egzersiz | Güven | Birincil Eklem Açısı | Tekrar (`find_peaks`) | ROM | Form Puanı ve Durumu |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `VID20261012132043.mp4` | Pec Deck / Machine Chest Fly (Ön) | %90.7 | `chest_fly_angle` | 4 Tekrar | 82.6° | 100 / 100 — Doğru Form (Yeşil İskelet) |
| `VID20261012132100.mp4` | Pec Deck / Machine Chest Fly (Yan) | %98.0 | `chest_fly_angle` | 2 Tekrar | 80.6° | 100 / 100 — Doğru Form (Yeşil İskelet) |
| `VID20261016125156.mp4` | Incline Chest Press Machine (Ön) | %94.0 | `elbow_angle` | 2 Tekrar | 75.9° | 100 / 100 — Doğru Form (Yeşil İskelet) |
| `VID20261016125216.mp4` | Incline Chest Press Machine (Yan) | %94.0 | `elbow_angle` | 3 Tekrar | 89.4° | 100 / 100 — Doğru Form (Yeşil İskelet) |
| `VID20261016132744.mp4` | Seated Dumbbell Shoulder Press | %94.5 | `shoulder_angle` | 3 Tekrar | 80.9° | 100 / 100 — Doğru Form (Yeşil İskelet) |
| `VID20261026140055.mp4` | Cable Triceps Pushdown (Yan) | %94.5 | `elbow_angle` | 4 Tekrar | 81.8° | 100 / 100 — Doğru Form (Yeşil İskelet) |
| `VID20261026140133.mp4` | Cable Triceps Pushdown (Yan) | %87.0 | `elbow_angle` | 2 Tekrar | 87.1° | 87.4 / 100 — Form Uyarısı (Bel Açısı < 150°) |

## 5. Proje Dosya Yapısı
* `kas_analiz_sistemi.py`: 3NF SQLite veritabanı yöneticisi, MediaPipe/Kinematik iskelet çıkarıcı, `atan2` açı hesaplayıcı, `scipy.signal.find_peaks` tekrar sayıcı, hareket sınıflandırıcı ve `threading` canlı kamera modülü.
* `plot_analysis_dashboard.py`: Veritabanındaki zaman serisi açı verilerini ve tepe/çukur noktalarını grafikleştiren görselleştirme betiği.
* `antrenman_analiz.db`: 3NF kurallarına uygun örnek SQLite veritabanı (`videos`, `frames`, `keypoints`, `angles`).
* `spor_algilama_zaman_serisi_analizi.png`: Örnek videoların zaman serisi ve tepe noktası analiz grafiği.
* `requirements.txt`: Gerekli Python kütüphaneleri.# Kas-Analiz-ve-Tekrar-Sayma-Sistemi
