from pathlib import Path

import cv2
from ultralytics import YOLO


class YOLOParkingDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.25):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))

    def predict(self, image):
        results = self.model(
            image,
            conf=self.confidence_threshold,
            verbose=False
        )

        result = results[0]

        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = result.names[class_id]

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            )

        return detections


def draw_yolo_detections(image, detections):
    result = image.copy()

    for det in detections:
        x1 = det["x1"]
        y1 = det["y1"]
        x2 = det["x2"]
        y2 = det["y2"]

        class_name = det["class_name"]
        confidence = det["confidence"]

        if class_name == "space-empty":
            color = (0, 255, 0)
            label = f"EMPTY {confidence:.2f}"
        elif class_name == "space-occupied":
            color = (0, 0, 255)
            label = f"OCC {confidence:.2f}"
        else:
            color = (0, 255, 255)
            label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            result,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

    return result


def calculate_yolo_statistics(detections):
    total = len(detections)

    free = sum(
        1 for det in detections
        if det["class_name"] == "space-empty"
    )

    occupied = sum(
        1 for det in detections
        if det["class_name"] == "space-occupied"
    )

    unknown = total - free - occupied

    occupancy_rate = occupied / total if total > 0 else 0

    return {
        "total_spots": total,
        "free_spots": free,
        "occupied_spots": occupied,
        "unknown_spots": unknown,
        "occupancy_rate": round(occupancy_rate, 3)
    }