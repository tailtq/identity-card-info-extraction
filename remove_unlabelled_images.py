import json
import os
import cv2
import numpy as np
import glob


def remove_images(images: list, keep_range_ids: tuple, dataset_path: str):
    # ImageSets/Segmentation/default.txt
    # JPEGImages/*.jpg
    # SegmentationClass/*.jpg
    # SegmentationObject/*.jpg
    default_file_path = f"{dataset_path}/ImageSets/Segmentation/default.txt"
    default_names = open(default_file_path, "r").read().split("\n")
    removing_images = []

    for image in images:
        if not (keep_range_ids[0] <= image["id"] <= keep_range_ids[1]):
            removing_images.append(image["file_name"])

    removing_names = [image.replace(".jpg", "") for image in removing_images]
    default_names = list(filter(lambda e: e not in removing_names, default_names))

    f = open(default_file_path, "w+")
    f.write("\n".join(default_names))
    f.close()

    for image in removing_images:
        jpg_image_path = f"{dataset_path}/JPEGImages/{image}"
        seg_class_path = f"{dataset_path}/SegmentationClass/{image.replace('.jpg', '.png')}"
        seg_obj_path = f"{dataset_path}/SegmentationObject/{image.replace('.jpg', '.png')}"

        if os.path.exists(jpg_image_path):
            os.remove(jpg_image_path)

        if os.path.exists(seg_class_path):
            os.remove(seg_class_path)

        if os.path.exists(seg_obj_path):
            os.remove(seg_obj_path)


def convert_rgb_to_indexed_colors(img, color_map, destination):
    img = cv2.cvtColor(img[:, :], cv2.COLOR_BGR2RGB)

    for index, color in enumerate(color_map):
        red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        mask = (red == color[0]) & (green == color[1]) & (blue == color[2])
        img[:, :, :3][mask] = [index, index, index]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(destination, img)


def get_color_map():
    color_map = []

    data = open("dataset/segmentation/labelmap.txt", "r").read()
    lines = data.split("\n")
    lines.pop(0)
    lines.pop(len(lines) - 1)

    for line in lines:
        color_map.append([int(color) for color in line.split(":")[1].split(",")])

    return np.array(color_map)


if __name__ == "__main__":
    # content = open("annotations.json", "r").read()
    # content = json.loads(content)

    # remove_images(content["images"], (1, 57), "dataset/segmentation")
    color_map = get_color_map()
    files = glob.glob("dataset/segmentation/SegmentationClass/*.png")

    for file in files:
        new_destination = file.replace("SegmentationClass", "SegmentationNewClass")
        img = cv2.imread(file)
        convert_rgb_to_indexed_colors(img, color_map, new_destination)
