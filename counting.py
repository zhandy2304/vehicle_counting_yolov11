import time
import cv2
from ultralytics import solutions

# Coba buka stream RTSP
rtsp_url = "rtsp://admin:dcttotal2019@36.67.188.241:558/LiveChannel/2/media.smp"
cap = cv2.VideoCapture(rtsp_url)  # Gunakan FFmpeg untuk kompatibilitas lebih baik

# Cek apakah stream berhasil dibuka
if not cap.isOpened():
    print("Error: Tidak dapat membuka stream RTSP. Periksa URL atau koneksi jaringan.")
    exit()

# Tunggu beberapa detik untuk memastikan stream siap
time.sleep(2)

# Ambil informasi video
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Resolusi: {w}x{h}, FPS: {fps}")

# Define region points untuk counting
region_points = [(60, 100), (360, 100), (360, 150), (60, 150)]  # Rectangle counting

# Init ObjectCounter dari Ultralytics
counter = solutions.ObjectCounter(
    show=True,  # Tampilkan output
    region=region_points,
    model="yolo11n.pt",  # Model YOLO
    show_out=False,  # Hanya hitung objek yang masuk
    line_width=1
)

# Loop pemrosesan video
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Frame tidak terbaca. Stream mungkin terputus atau selesai.")
        break

    # Hitung objek dengan YOLO
    frame = counter.count(frame)

    # Tampilkan video dengan OpenCV
    cv2.imshow("RTSP Stream", frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resource setelah selesai
cap.release()
cv2.destroyAllWindows()
