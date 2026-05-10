import json
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parents[1]

IMAGE_PATH = BASE_DIR / "test_parking.jpg"

OUTPUT_JSON = BASE_DIR / "configs" / "parking_config_generated.json"


spots = []

drawing = False

start_x = 0
start_y = 0

current_image = None
original_image = None


def redraw_image():
    global current_image

    current_image = original_image.copy()

    for spot in spots:

        x1 = spot["x1"]
        y1 = spot["y1"]
        x2 = spot["x2"]
        y2 = spot["y2"]

        cv2.rectangle(
            current_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            current_image,
            str(spot["id"]),
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )


def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_x
    global start_y
    global current_image

    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        start_x = x
        start_y = y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:

        temp_image = current_image.copy()

        cv2.rectangle(
            temp_image,
            (start_x, start_y),
            (x, y),
            (255, 255, 0),
            2
        )

        cv2.imshow("Parking Spot Editor", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        x1 = min(start_x, x)
        y1 = min(start_y, y)

        x2 = max(start_x, x)
        y2 = max(start_y, y)

        spot = {
            "id": len(spots) + 1,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }

        spots.append(spot)

        redraw_image()

        cv2.imshow("Parking Spot Editor", current_image)

        print(f"Добавлено место #{spot['id']}")


def save_config():

    config = {
        "parking_id": "generated_parking",
        "description": "Generated parking configuration",
        "spots": spots
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:

        json.dump(
            config,
            f,
            indent=4
        )

    print("\nКонфиг сохранён:")
    print(OUTPUT_JSON)


def main():

    global current_image
    global original_image

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:

        print("Ошибка загрузки изображения")
        return

    original_image = image
    current_image = image.copy()

    cv2.namedWindow("Parking Spot Editor")

    cv2.setMouseCallback(
        "Parking Spot Editor",
        mouse_callback
    )

    print("\nИНСТРУКЦИЯ:")
    print("ЛКМ + движение = создать место")
    print("S = сохранить")
    print("Q = выход")

    while True:

        cv2.imshow(
            "Parking Spot Editor",
            current_image
        )

        key = cv2.waitKey(1)

        if key == ord("s"):

            save_config()

        elif key == ord("q"):

            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()