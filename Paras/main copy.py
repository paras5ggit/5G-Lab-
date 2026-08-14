import cv2
import time 
from ultralytics import YOLO

# Waste model
waste_model = YOLO("runs/detect/train-2/weights/best.pt")

# Person model
person_model = YOLO("yolov8n.pt")

# RTSP Camera URL
rtsp_url = "rtsp://192.168.30.23:8554/primary1"

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Cannot connect to camera")
    exit()

frame_count = 0

while True:
    cap.grab()
    ret, frame = cap.retrieve()
    frame_count += 1 
    if frame_count % 3 !=0:
        continue

    if not ret:
        print("Frame lost... reconnecting")
        continue
    frame = cv2.resize(frame, (640, 480))

    # Waste detection
    start_time = time.time()

    waste_results = waste_model.predict(
    frame,
    conf=0.45,
    iou=0.60,
    imgsz=640,
    verbose=True
    )

    waste_time = (time.time() - start_time) *1000 
    print(f"Waste detection time: {waste_time:.2f} ms") 

    # Person detection
    person_results = person_model.predict(
    frame,
    classes=[0], # 0 = person
    conf=0.50,
    verbose=True
    )

    display = frame.copy()

    # Draw waste detections (RED)
    for r in waste_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = f"{waste_model.names[cls]} {conf:.2f}"

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(display, label,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,255),
                    2)

    # Draw person detections (GREEN)
    for r in person_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            label = f"Person {conf:.2f}"

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)


    cv2.imshow("AI Waste Detection", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
