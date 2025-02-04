import cv2
import os

# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000"

cap = cv2.VideoCapture("rtsp://admin:dcttotal2019@36.67.188.241:558/LiveChannel/8/media.smp")

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()