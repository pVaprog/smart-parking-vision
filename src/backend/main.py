from src.model.yolo_detector import (
    YOLOParkingDetector,
    draw_yolo_detections,
    calculate_yolo_statistics
)

from pathlib import Path
import uuid

import random
import time

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse

from src.parking.spot_config import load_parking_config
from src.parking.occupancy_detector import mock_detect_occupancy, calculate_statistics
from src.parking.visualizer import draw_parking_spots


app = FastAPI(
    title="Smart Parking Vision API",
    description="AI Parking Occupancy Detection System",
    version="1.0"
)

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "configs" / "parking_config_generated.json"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = BASE_DIR / "models" / "parking_yolov8n_best.pt"

yolo_detector = YOLOParkingDetector(
    model_path=str(MODEL_PATH),
    confidence_threshold=0.25
)

@app.get("/")
def root():
    return {
        "message": "Smart Parking Vision API работает"
    }


@app.get("/health")
def health_check():
    return {
        "status": "OK"
    }


@app.get("/parking-config")
def get_parking_config():
    try:
        config = load_parking_config(str(CONFIG_PATH))
        return config

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        np_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Не удалось прочитать изображение"}
            )

        height, width, channels = image.shape

        return {
            "filename": file.filename,
            "width": width,
            "height": height,
            "channels": channels,
            "message": "Изображение успешно загружено"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/analyze-parking")
async def analyze_parking(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        np_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Не удалось прочитать изображение"}
            )

        config = load_parking_config(str(CONFIG_PATH))
        spots = config["spots"]

        detected_spots = mock_detect_occupancy(spots)
        statistics = calculate_statistics(detected_spots)

        visualized_image = draw_parking_spots(
            image,
            detected_spots,
            statistics
        )

        output_filename = f"parking_result_{uuid.uuid4().hex}.jpg"
        output_path = OUTPUT_DIR / output_filename

        cv2.imwrite(str(output_path), visualized_image)

        return {
            "parking_id": config["parking_id"],
            "statistics": statistics,
            "spots": detected_spots,
            "result_image": f"/result-image/{output_filename}"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/result-image/{filename}")
def get_result_image(filename: str):
    image_path = OUTPUT_DIR / filename

    if not image_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Файл результата не найден"}
        )

    return FileResponse(str(image_path))

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):

    try:

        video_bytes = await file.read()

        temp_video_path = OUTPUT_DIR / f"temp_{uuid.uuid4().hex}.mp4"

        with open(temp_video_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(str(temp_video_path))

        if not cap.isOpened():

            return JSONResponse(
                status_code=400,
                content={
                    "error": "Не удалось открыть видео"
                }
            )

        config = load_parking_config(str(CONFIG_PATH))
        spots = config["spots"]

        processed_frames = 0

        free_history = []
        occupied_history = []

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            processed_frames += 1

            if processed_frames % 10 != 0:
                continue

            simulated_spots = []

            for spot in spots:

                spot_copy = spot.copy()

                random_value = random.random()

                if random_value > 0.5:

                    spot_copy["status"] = "occupied"

                else:

                    spot_copy["status"] = "free"

                simulated_spots.append(spot_copy)

            stats = calculate_statistics(simulated_spots)

            free_history.append(stats["free_spots"])
            occupied_history.append(stats["occupied_spots"])

            time.sleep(0.03)

        cap.release()

        return {
            "video_processed": True,
            "processed_frames": processed_frames,
            "free_history": free_history,
            "occupied_history": occupied_history
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )
        
@app.post("/analyze-parking-yolo")
async def analyze_parking_yolo(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        np_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Не удалось прочитать изображение"}
            )

        detections = yolo_detector.predict(image)

        statistics = calculate_yolo_statistics(detections)

        visualized_image = draw_yolo_detections(
            image,
            detections
        )

        output_filename = f"parking_yolo_result_{uuid.uuid4().hex}.jpg"
        output_path = OUTPUT_DIR / output_filename

        cv2.imwrite(str(output_path), visualized_image)

        return {
            "mode": "yolo",
            "model": str(MODEL_PATH.name),
            "statistics": statistics,
            "detections_count": len(detections),
            "result_image": f"/result-image/{output_filename}"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )