"""
After using an annotation tool to label your images, export your labels to YOLO format, with one *.txt file per image (if no objects in image, no *.txt file is required). The *.txt file specifications are:

* One row per object
* Each row is `class x_center y_center width height` format.
* Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide `x_center` and `width` by image width, and `y_center` and `height` by image height.
* Class numbers are zero-indexed (start from 0).
"""
from pathlib import Path
import json
from collections import defaultdict
import os

def generate_yolo_labels(dataset_path, folder):
    os.mkdir(dataset_path / "labels" / folder)

    ann_path = dataset_path / "images" / folder / "_annotations.coco.json"
    data = json.loads(ann_path.read_text())

    categories = data['categories']
    images = data['images']

    # Create a dict that maps image_id to file name 
    image_id_to_name = {}
    for image in images:
        image_id_to_name[image['id']] = image['file_name']

    # Group the annotations by their image_id
    # i.e. convert to a dictionary, where (key, value) = (image_id, list of annotations)
    annotations_dict = defaultdict(list)
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        if image_id in annotations_dict:
            annotations_dict[image_id].append(annotation)
        else:
            annotations_dict[image_id] = [annotation]

    # Delete those images that have no annotations
    images = [img for img in images if img['id'] in annotations_dict]

    # For each image, create a *.txt file
    label_folder_path = dataset_path / "labels" / folder
    print(f"Generating labels for {folder} data. Save to {label_folder_path}")
    for image in images:
        image_id = image['id']
        image_name = image_id_to_name[image_id]
        if image_name.endswith(".jpg"):
            image_name = image_name[:-4] + ".txt"
        else:
            raise ValueError(f"Image name {image_name} does not end with .jpg")
        
        image_annotations = annotations_dict[image_id]
        image_path = label_folder_path / image_name
        with open(image_path, "w") as f:
            for annotation in image_annotations:
                box = annotation['bbox']
                class_id = annotation['category_id']
                # Convert box coordinates (xmin, ymin, w, h) to normalized xywh format
                x_center = (box[0] + box[2] / 2) / image['width']
                y_center = (box[1] + box[3] / 2) / image['height']
                width = box[2] / image['width']
                height = box[3] / image['height']
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def main():
    folders = ["train", "valid"]
    dataset_path = Path(r"hw1_dataset")

    for folder in folders:
        generate_yolo_labels(dataset_path, folder)


if __name__ == "__main__":
    main()