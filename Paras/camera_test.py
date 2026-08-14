import cv2

rtsp_url = "rtsp://192.168.30.23:8554/primary1"

cap = cv2.VideoCapture(rtsp_url)

print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    print("Cannot connect to camera")
    exit()

while True:
    ret, frame = cap.read()

    print("ret =", ret)

    if not ret:
        print("Frame Lost")
        break

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
