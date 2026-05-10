import json
from pathlib import Path


def load_parking_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)