from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="configs/pklot.yaml",
    epochs=5,
    imgsz=416,
    batch=4,
    workers=2,
    device="cpu"
)