from ultralytics import YOLO

# Load the pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train the model on the waste-management dataset
results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="cpu",
    pretrained=True
)
