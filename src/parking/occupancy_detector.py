def mock_detect_occupancy(spots):
    result_spots = []

    for spot in spots:
        spot_copy = spot.copy()

        if spot_copy["id"] % 2 == 0:
            spot_copy["status"] = "occupied"
            spot_copy["confidence"] = 0.87
        else:
            spot_copy["status"] = "free"
            spot_copy["confidence"] = 0.91

        result_spots.append(spot_copy)

    return result_spots


def calculate_statistics(spots):
    total = len(spots)
    free = sum(1 for spot in spots if spot["status"] == "free")
    occupied = sum(1 for spot in spots if spot["status"] == "occupied")
    unknown = total - free - occupied

    occupancy_rate = occupied / total if total > 0 else 0

    return {
        "total_spots": total,
        "free_spots": free,
        "occupied_spots": occupied,
        "unknown_spots": unknown,
        "occupancy_rate": round(occupancy_rate, 3)
    }