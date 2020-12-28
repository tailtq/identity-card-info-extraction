import os
import cv2
import torch
import numpy as np

from remove_unlabelled_images import get_color_map
from utils.object_recognition_common import warp_identity_card
from utils.ocr_common import OCRCommon
from utils.segmentation_common import get_validation_augmentation, get_preprocessing, \
    get_preprocessing_fn, get_min_max_x_y
from utils.object_recognition_common import load_model as load_warping_model
from utils.segmentation_common import load_model as load_segmentation_model

DEVICE = "cuda"
ocr_model = OCRCommon()
warping_model = load_warping_model("warping_model.pt")
segmentation_model = load_segmentation_model("info_segmentation.pth")

segmentation_preprocessing = get_validation_augmentation()
segmentation_augmentation = get_preprocessing(get_preprocessing_fn())

DATA_DIR = "./dataset/segmentation/"
x_test_dir = os.path.join(DATA_DIR, "test")
y_test_dir = os.path.join(DATA_DIR, "testannot")
original_size = [500, 300]
resized_size = [480, 480]

color_map, labels = get_color_map(True)
color_map, labels = color_map[1:], labels[1:]
converted_color_map = np.repeat(np.repeat(color_map[:, :, np.newaxis, np.newaxis], 480, axis=2), 480, axis=3)

if __name__ == "__main__":
    # test dataset without transformations for image visualization
    img_path = "dataset/Java_Le Van Huy.JPG"

    img_name = img_path.split('/')[-1]
    img = cv2.imread(img_path)
    img = warp_identity_card(img, warping_model)
    img = cv2.resize(img, (480, 480))
    orig_img = img.copy()

    sample = segmentation_augmentation(image=img, mask=np.zeros((480, 480, 3), dtype=np.uint8))
    img = sample["image"]
    sample = segmentation_preprocessing(image=img, mask=np.zeros((480, 480, 3), dtype=np.uint8))
    img = sample["image"]

    # segment information
    x_tensor = torch.from_numpy(img).to(DEVICE).unsqueeze(0)
    pr_mask = segmentation_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    pr_mask = np.repeat(pr_mask[:, np.newaxis, :, :], 3, axis=1) * converted_color_map
    pr_mask = np.sum(pr_mask, axis=0).transpose((1, 2, 0)).astype(np.uint8)

    # extract information by each color
    for index, color in enumerate(color_map):
        indices = np.where(np.all(pr_mask == color, axis=-1))
        coords = np.array(list(zip(indices[0], indices[1])))

        if len(coords) == 0:
            continue

        min_x, min_y, max_x, max_y = get_min_max_x_y(coords)
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        white_area = np.ones((height, width, 3), dtype=np.uint8) * 255
        white_area[coords[:, 0] - min_y, coords[:, 1] - min_x] = orig_img[coords[:, 0], coords[:, 1]]

        # split large bbox if height > 70 (just estimated)
        if height > 70:
            half_height = int(height / 2)
            white_area1 = white_area[0:half_height, :, :]
            white_area2 = white_area[half_height:, :, :]

            text = ocr_model.predict(white_area1)
            text += " " + ocr_model.predict(white_area2)

            cv2.imshow("Test", white_area1)
            cv2.waitKey(-1)
            cv2.imshow("Test", white_area2)
            cv2.waitKey(-1)
        else:
            text = ocr_model.predict(white_area)

            cv2.imshow("Test", white_area)
            cv2.waitKey(-1)

        print(f"{labels[index]['name']}: {text}")

    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2BGRA)
    pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_BGR2BGRA)
    orig_img[pr_mask > 0] += (pr_mask[pr_mask > 0] / 4).astype(np.uint8)

    cv2.imshow("Test", orig_img)
    cv2.waitKey(-1)
