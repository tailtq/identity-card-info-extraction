import os
import cv2
import torch
import numpy as np
import json

from remove_unlabelled_images import get_color_map
from utils.segmentation_common import get_validation_augmentation, get_preprocessing, load_model, CATEGORIES, \
    CocoDataset, get_preprocessing_fn, get_min_max_x_y, get_segment

DEVICE = "cuda"
segmentation_model = load_model("info_segmentation.pth")
original_size = [500, 300]
resized_size = [480, 480]


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


if __name__ == '__main__':
    start_id = 901
    start_image_id = 202

    json_data = json.loads(open("instances_default.json", "r").read())
    # test dataset without transformations for image visualization
    test_dataset_vis = CocoDataset(
        json_data,
        "dataset/segmentation_input",
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(get_preprocessing_fn()),
        classes=CATEGORIES,
        from_id=start_image_id,
    )
    color_map, categories = get_color_map(True)
    color_map, categories = color_map[1:], categories[1:]

    color_map_shape = color_map.shape
    new_color_map = color_map.reshape((color_map_shape[0], color_map_shape[1], 1, 1))
    json_annotations = []

    for image, gt_mask in test_dataset_vis:
        image_height, image_width, _ = image.shape
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

        pr_mask = segmentation_model.predict(x_tensor)
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
            area = float(width * height)
            bbox = np.array([
                float(min_x),
                float(min_y),
                float(max_x - min_x),
                float(max_y - min_y)
            ])

            # split large bbox if height > 70 (just estimated)
            if height > 70:
                middle_y = min_y + int(height / 2 - 5)

                # split into 2 coordinate types
                cut_coords_above = coords[np.argwhere(coords[:, 0] <= middle_y).reshape(-1)]
                cut_coords_below = coords[np.argwhere(coords[:, 0] > middle_y).reshape(-1)]
                min_x_a, min_y_a, max_x_a, max_y_a = get_min_max_x_y(cut_coords_above)
                min_x_b, min_y_b, max_x_b, max_y_b = get_min_max_x_y(cut_coords_below)
                segment_area = get_segment(min_x_a, min_y_a, max_x_a, max_y_a, min_x_b, min_y_b, max_x_b, max_y_b)
                area = float((max_x_a - min_x_a) * (max_y_a - min_y_a) + (max_x_b - min_x_b) * (max_y_b - min_y_b))
            else:
                segment_area = np.array([
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y],
                ])

            bbox[0] *= original_size[0] / resized_size[0]
            bbox[1] *= original_size[1] / resized_size[1]
            bbox[2] *= original_size[0] / resized_size[0]
            bbox[3] *= original_size[1] / resized_size[1]
            segment_area[:, 0] = segment_area[:, 0] * original_size[0] / resized_size[0]
            segment_area[:, 1] = segment_area[:, 1] * original_size[1] / resized_size[1]
            # draw_segmentation(pr_mask, segment_area)
            segment = get_coco_format(start_id,  # missing piece
                                      start_image_id,  # missing piece
                                      categories[index]["id"],
                                      segment_area.reshape(segment_area.size,).astype(float).tolist(),
                                      bbox.tolist(),
                                      area)
            json_annotations.append(segment)
            start_id += 1

        start_image_id += 1
        # cv2.imshow("Test", pr_mask)
        # key = cv2.waitKey(-1)
        #
        # if key == ord('q'):
        #     break

    f = open("test.json", "w+")
    f.write(json.dumps(json_annotations))
    f.close()
