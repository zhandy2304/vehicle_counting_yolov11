import cv2
import time
import mysql.connector
from ultralytics import YOLO
from collections import defaultdict

# Konfigurasi database
# conn = mysql.connector.connect(
#     host='localhost',
#     user='root',
#     password='your_password',
#     database='your_database'
# )
# cursor = conn.cursor()

# Buat tabel jika belum ada
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS vehicle_count (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     class VARCHAR(50),
#     timestamp DATETIME
# )
# ''')
# conn.commit()

# Load model YOLO
model = YOLO('yolov8n.pt')  # Pastikan model tersedia di direktori yang benar

# Buka video atau webcam
cap = cv2.VideoCapture('videos/traffic video.mp4')  # Ganti dengan 0 untuk webcam

# Posisi garis deteksi
line_y = 150

# Dictionary untuk menghitung jumlah kendaraan
vehicle_count = defaultdict(int)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_width = frame.shape[1]
    line_x1 = 200
    line_x2 = frame_width - 200
    
    # Deteksi objek
    results = model(frame)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            center_y = (y1 + y2) / 2
            
            # Jika kendaraan melewati garis
            if center_y > line_y - 5 and center_y < line_y + 5:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                # cursor.execute('INSERT INTO vehicle_count (class, timestamp) VALUES (%s, %s)', (class_name, timestamp))
                # conn.commit()
                print(f'{class_name} melewati garis pada {timestamp}')
                vehicle_count[class_name] += 1
                
                # Gambar kotak dan garis
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Gambar garis deteksi
    cv2.line(frame, (line_x1, line_y), (line_x2, line_y), (0, 0, 255), 2)
    
    # Tampilkan jumlah kendaraan di kanan atas
    y_offset = 20
    for class_name, count in vehicle_count.items():
        cv2.putText(frame, f'{class_name}: {count}', (frame_width - 200, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 20
    
    # Tampilkan frame
    cv2.imshow('Vehicle Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup koneksi dan video
cap.release()
# cursor.close()
# conn.close()
cv2.destroyAllWindows()
