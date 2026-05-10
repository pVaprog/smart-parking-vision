from ultralytics import YOLO

model = YOLO("runs/detect/train-2/weights/best.pt")

model.predict(
    source="data/test",
    save=True,
    conf=0.25
)