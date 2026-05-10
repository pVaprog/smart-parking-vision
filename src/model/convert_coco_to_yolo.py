import json
import shutil
from pathlib import Path


DATA_DIR = Path("data")

SPLITS = ["train", "valid", "test"]


def coco_bbox_to_yolo(bbox, img_width, img_height):
    """
    COCO bbox: [x_min, y_min, width, height]
    YOLO bbox: x_center y_center width height, all normalized 0..1
    """
    x_min, y_min, box_w, box_h = bbox

    x_center = x_min + box_w / 2
    y_center = y_min + box_h / 2

    x_center /= img_width
    y_center /= img_height
    box_w /= img_width
    box_h /= img_height

    return x_center, y_center, box_w, box_h


def prepare_split(split_name):
    split_dir = DATA_DIR / split_name
    annotation_path = split_dir / "_annotations.coco.json"

    if not annotation_path.exists():
        print(f"[ОШИБКА] Нет файла: {annotation_path}")
        return

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    print(f"\nОбработка папки: {split_name}")

    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    print("Категории в COCO:")
    for cat in categories:
        print(cat)

    # Сопоставляем image_id -> информация об изображении
    image_id_to_info = {}

    for img in images:
        image_id_to_info[img["id"]] = img

    # Сопоставляем image_id -> список аннотаций
    image_id_to_annotations = {}

    for ann in annotations:
        image_id = ann["image_id"]

        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []

        image_id_to_annotations[image_id].append(ann)

    # ВАЖНО:
    # Roboflow часто делает category_id не с 0, а с 1.
    # YOLO требует классы с 0.
    category_ids = sorted([cat["id"] for cat in categories])
    category_id_to_yolo_id = {
        category_id: index for index, category_id in enumerate(category_ids)
    }

    converted_images = 0
    converted_labels = 0

    for image_id, img_info in image_id_to_info.items():
        file_name = img_info["file_name"]
        img_width = img_info["width"]
        img_height = img_info["height"]

        old_image_path = split_dir / file_name
        new_image_path = images_dir / Path(file_name).name

        # Переносим jpg в папку images
        if old_image_path.exists() and not new_image_path.exists():
            shutil.move(str(old_image_path), str(new_image_path))

        # Если файл уже лежит в images, это нормально
        if not new_image_path.exists():
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Не найдено изображение: {new_image_path}")
            continue

        label_file = labels_dir / (Path(file_name).stem + ".txt")

        image_annotations = image_id_to_annotations.get(image_id, [])

        yolo_lines = []

        for ann in image_annotations:
            category_id = ann["category_id"]

            if category_id not in category_id_to_yolo_id:
                continue

            class_id = category_id_to_yolo_id[category_id]

            bbox = ann["bbox"]

            x_center, y_center, box_w, box_h = coco_bbox_to_yolo(
                bbox,
                img_width,
                img_height
            )

            # На всякий случай ограничиваем значения
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            box_w = max(0, min(1, box_w))
            box_h = max(0, min(1, box_h))

            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
            yolo_lines.append(yolo_line)

        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        converted_images += 1
        converted_labels += len(yolo_lines)

    print(f"Изображений обработано: {converted_images}")
    print(f"Объектов-разметок обработано: {converted_labels}")


def main():
    for split in SPLITS:
        prepare_split(split)

    print("\nГотово! COCO-разметка конвертирована в YOLO-формат.")


if __name__ == "__main__":
    main()