import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import time

from remove_unlabelled_images import get_color_map
from utils.segmentation_common import Dataset, get_validation_augmentation, get_preprocessing, load_model, CATEGORIES

ENCODER = 'vgg13_bn'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

DEVICE = 'cuda'

best_model = load_model()
DATA_DIR = './dataset/segmentation/'
x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')


# yx
def get_min_max_x_y(coordinates):
    min_y, min_x, max_y, max_x = min(coordinates[:, 0]), min(coordinates[:, 1]), max(coordinates[:, 0]), max(coordinates[:, 1])

    return min_x, min_y, max_x, max_y


def draw_vertex(min_x, min_y, max_x, max_y, color):
    cv2.circle(pr_mask, (min_x, min_y), 4, color, -1)
    cv2.circle(pr_mask, (max_x, min_y), 4, color, -1)
    cv2.circle(pr_mask, (max_x, max_y), 4, color, -1)
    cv2.circle(pr_mask, (min_x, max_y), 4, color, -1)


# xy
def draw_segmentation(img, coordinates):
    length = len(coordinates)

    for i in range(length):
        coordinate1 = coordinates[i]
        coordinate2 = coordinates[(i + 1) % length]

        cv2.line(img,
                 (coordinate1[0], coordinate1[1]),
                 (coordinate2[0], coordinate2[1]),
                 (0, 255, 0),
                 2)


def get_coco_format(id, img_id, category_id, segmentation, bbox, area):
    return {
        "id": id,
        "image_id": img_id,
        "category_id": category_id,
        "segmentation": [segmentation],
        "area": area,  # need to calculate
        "bbox": bbox,
        "iscrowd": 0,
        "attributes": {
            "occluded": False
        }
    }


# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CATEGORIES,
)
color_map, categories = get_color_map(True)
color_map, categories = color_map[1:], categories[1:]

color_map_shape = color_map.shape
new_color_map = color_map.reshape((color_map_shape[0], color_map_shape[1], 1, 1))
json_annotations = []

for image, gt_mask in test_dataset_vis:
    image_height, image_width, _ = image.shape
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    pr_mask = np.repeat(pr_mask[:, np.newaxis, :, :], 3, axis=1) * new_color_map
    pr_mask = np.sum(pr_mask, axis=0).transpose((1, 2, 0)).astype(np.uint8)
    # bug if not converting    Expected Ptr<cv::UMat> for argument 'img'
    pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_RGB2BGR)

    for index, color in enumerate(color_map):
        indices = np.where(np.all(pr_mask == np.flip(color), axis=-1))
        coords = np.array(list(zip(indices[0], indices[1])))

        if len(coords) == 0:
            continue

        min_x, min_y, max_x, max_y = get_min_max_x_y(coords)
        width = max_x - min_x
        height = max_y - min_y
        draw_vertex(min_x, min_y, max_x, max_y, color=(255, 255, 255))

        area = width * height
        bbox = [
            min_x / image_width,
            min_y / image_height,
            (max_x - min_x) / image_width,
            (max_y - min_y) / image_height
        ]

        # split large bbox if height > 70 (just estimated)
        if height > 70:
            middle_y = min_y + int(height / 2 - 7)

            # split into 2 coordinate types
            cut_coords_above = coords[np.argwhere(coords[:, 0] <= middle_y).reshape(-1)]
            cut_coords_below = coords[np.argwhere(coords[:, 0] > middle_y).reshape(-1)]
            min_x_a, min_y_a, max_x_a, max_y_a = get_min_max_x_y(cut_coords_above)
            min_x_b, min_y_b, max_x_b, max_y_b = get_min_max_x_y(cut_coords_below)

            segmentation = np.array([
                [min_x_a, min_y_a],
                [max_x_a, min_y_a],
                [max_x_a, max_y_a],
                [max_x_b, min_y_b],
                [max_x_b, max_y_b],
                [min_x_b, max_y_b],
                [min_x_b, min_y_b],
                [min_x_a, max_y_a],
            ])
            area = (max_x_a - min_x_a) * (max_y_a - min_y_a) + (max_x_b - min_x_b) * (max_y_b - min_y_b)
        else:
            segmentation = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
            ])

        draw_segmentation(pr_mask, segmentation)
        get_coco_format(id,  # missing piece
                        img_id,  # missing piece
                        categories[index],
                        segmentation,
                        bbox,
                        area)

    cv2.imshow("Test", pr_mask)
    key = cv2.waitKey(-1)

    if key == ord('q'):
        break
