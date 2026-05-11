from pathlib import Path

import cv2
from ultralytics import YOLO

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


class YOLOParkingDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.35):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))

        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(self.model_path),
            confidence_threshold=self.confidence_threshold,
            device="cpu"
        )

    def predict(self, image):
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=0.35,
            imgsz=1920,
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

    def predict_tiled(self, image):
        result = get_sliced_prediction(
            image,
            self.sahi_model,
            slice_height=135,
            slice_width=135,
            overlap_height_ratio=0.35,
            overlap_width_ratio=0.35,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.35,
            verbose=0
        )

        detections = []

        for obj in result.object_prediction_list:
            bbox = obj.bbox

            class_id = int(obj.category.id)
            class_name = obj.category.name
            confidence = float(obj.score.value)

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "x1": int(bbox.minx),
                    "y1": int(bbox.miny),
                    "x2": int(bbox.maxx),
                    "y2": int(bbox.maxy)
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

        if class_name == "space-empty":
            color = (0, 255, 0)
            label = "FREE"
        elif class_name == "space-occupied":
            color = (0, 0, 255)
            label = "OCC"
        else:
            color = (0, 255, 255)
            label = class_name

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            1
        )

        cv2.putText(
            result,
            label,
            (x1, max(y1 - 3, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
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