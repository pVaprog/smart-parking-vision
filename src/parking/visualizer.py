import cv2


def draw_statistics_panel(image, statistics):

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (320, 140),
        (40, 40, 40),
        -1
    )

    alpha = 0.75

    cv2.addWeighted(
        overlay,
        alpha,
        image,
        1 - alpha,
        0,
        image
    )

    total = statistics["total_spots"]
    free = statistics["free_spots"]
    occupied = statistics["occupied_spots"]

    occupancy_rate = statistics["occupancy_rate"]

    lines = [
        f"Total spots: {total}",
        f"Free spots: {free}",
        f"Occupied spots: {occupied}",
        f"Occupancy: {occupancy_rate * 100:.1f}%"
    ]

    y = 40

    for line in lines:

        cv2.putText(
            image,
            line,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        y += 25

    cv2.rectangle(
        image,
        (25, 115),
        (45, 135),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        image,
        "FREE",
        (55, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.rectangle(
        image,
        (140, 115),
        (160, 135),
        (0, 0, 255),
        -1
    )

    cv2.putText(
        image,
        "OCCUPIED",
        (170, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return image


def draw_parking_spots(image, spots, statistics=None):

    result = image.copy()

    for spot in spots:

        x1 = int(spot["x1"])
        y1 = int(spot["y1"])
        x2 = int(spot["x2"])
        y2 = int(spot["y2"])

        status = spot.get("status", "unknown")

        if status == "free":

            color = (0, 255, 0)

        elif status == "occupied":

            color = (0, 0, 255)

        else:

            color = (0, 255, 255)

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

    if statistics is not None:

        result = draw_statistics_panel(
            result,
            statistics
        )

    return result