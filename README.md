# Smart Parking Vision

**Smart Parking Vision** — это AI-система для анализа занятости парковочных мест по изображению или видео с камеры.  
Проект объединяет компьютерное зрение, backend API, веб-интерфейс и YOLO-модель для определения свободных и занятых парковочных мест.

## Основная идея проекта

Система принимает изображение парковки, анализирует его с помощью обученной YOLO-модели и возвращает:

- количество найденных парковочных мест;
- количество свободных мест;
- количество занятых мест;
- процент загрузки парковки;
- изображение с визуальной разметкой;
- аналитику по видео.

Проект может использоваться как MVP для:

- кампусов;
- торговых центров;
- офисных парковок;
- бизнес-центров;
- систем “умного города”.

## AI-модель

В проекте используется модель **YOLOv8**, обученная на датасете парковочных мест.

Классы модели:

```text
0 — space-empty
1 — space-occupied
```
Модель определяет на изображении отдельные парковочные места и классифицирует их как свободные или занятые.

## Датасет

Для обучения использовался датасет парковочных мест в COCO-формате:

data/
├── train/
├── valid/
└── test/

Каждая папка содержит изображения и файл:

_annotations.coco.json

Перед обучением датасет был преобразован из COCO-формата в YOLO-формат:

data_yolo/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

## Notebook обучения модели

Обучение модели выполнялось в Google Colab с использованием GPU Tesla T4.

Colab Notebook:

notebooks/Smart_Parking_YOLO_Training.ipynb

Если notebook загружен в GitHub, ссылка будет выглядеть так:

https://github.com/USERNAME/smart-parking-vision/blob/main/notebooks/Smart_Parking_YOLO_Training.ipynb

Замените USERNAME на ваш GitHub-логин.

## Структура проекта
smart-parking-vision/
│
├── app/
│   └── streamlit_app.py
│
├── configs/
│   ├── parking_config_generated.json
│   └── pklot_yolo.yaml
│
├── models/
│   └── parking_yolov8n_best.pt
│
├── notebooks/
│   └── Smart_Parking_YOLO_Training.ipynb
│
├── outputs/
│
├── src/
│   ├── backend/
│   │   └── main.py
│   │
│   ├── model/
│   │   └── yolo_detector.py
│   │
│   └── parking/
│       ├── occupancy_detector.py
│       ├── spot_config.py
│       └── visualizer.py
│
├── tools/
│   └── parking_spot-editor.py
│
├── requirements.txt
└── README.md

## Технологии

Проект реализован на Python.
Используются:

Python
YOLOv8 / Ultralytics
OpenCV
FastAPI
Streamlit
PyTorch
Pillow
Requests
Google Colab
GitHub

## Установка проекта
1. Клонировать репозиторий
git clone https://github.com/USERNAME/smart-parking-vision.git
cd smart-parking-vision
2. Создать виртуальное окружение

Windows:

python -m venv .venv
.venv\Scripts\activate

Linux / macOS:

python3 -m venv .venv
source .venv/bin/activate
3. Установить зависимости
pip install -r requirements.txt
4. Добавить обученную модель

Создайте папку:

models/

Положите туда файл модели:

parking_yolov8n_best.pt

Итоговая структура должна быть:

models/
└── parking_yolov8n_best.pt

## Запуск проекта

Проект запускается двумя отдельными процессами:

1. Запуск backend

В первом терминале:

uvicorn src.backend.main:app --reload

После запуска backend будет доступен по адресу:

http://127.0.0.1:8000

Swagger-документация API:

http://127.0.0.1:8000/docs

2. Запуск frontend

Во втором терминале:

streamlit run app/streamlit_app.py

После запуска интерфейс будет доступен по адресу:

http://localhost:8501
🧪 Как проверить работу
Запустите backend.
Запустите Streamlit frontend.
Откройте http://localhost:8501.
Загрузите изображение парковки.
Нажмите Analyze parking.
Система покажет:
количество найденных мест;
свободные места;
занятые места;
процент загрузки;
изображение с AI-разметкой.

## Обучение модели

Модель обучалась в Google Colab.

Основные этапы обучения:

подключение Google Drive;
проверка GPU;
загрузка датасета;
конвертация COCO → YOLO;
создание YAML-конфига;
обучение YOLOv8;
валидация модели;
сохранение best.pt.

Пример запуска обучения:

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="configs/pklot_yolo.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    device=0
)
## Результаты

После обучения YOLO формирует:

runs/
├── confusion_matrix.png
├── results.png
├── val_batch0_pred.jpg
└── weights/
    └── best.pt

Эти файлы позволяют оценить качество модели и показать результаты на защите.

## Особенности MVP

В проекте реализовано:

полноценное FastAPI API;
Streamlit dashboard;
загрузка изображений;
анализ видео;
YOLO inference;
визуализация результатов;
подсчёт свободных и занятых мест;
подготовленный pipeline обучения модели;
Colab Notebook для воспроизводимого обучения.

## Возможности развития

В дальнейшем проект можно расширить:

подключить live RTSP/IP-камеру;
добавить карту парковки;
добавить историю загрузки парковки;
сделать Telegram-бота;
добавить авторизацию;
развернуть backend в Docker;
подключить PostgreSQL;
сделать real-time monitoring.

## Автор

Проект выполнен в рамках студенческого AI-стартапа.

Название проекта: Smart Parking Vision
Тип проекта: Computer Vision / Smart City / Parking Analytics
Основная модель: YOLOv8
Backend: FastAPI
Frontend: Streamlit