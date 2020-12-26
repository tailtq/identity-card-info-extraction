import os
import cv2
import torch
import numpy as np

from remove_unlabelled_images import get_color_map
from utils.segmentation_common import Dataset, get_validation_augmentation, get_preprocessing, load_model, CATEGORIES, \
    get_preprocessing_fn

DEVICE = 'cuda'

best_model = load_model()
DATA_DIR = './dataset/segmentation/'
x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(get_preprocessing_fn()),
    classes=CATEGORIES,
)
color_map = get_color_map()[1:]
converted_color_map = np.repeat(np.repeat(color_map[:, :, np.newaxis, np.newaxis], 480, axis=2), 480, axis=3)

for i, (image, gt_mask) in enumerate(test_dataset_vis):
    orig_image = cv2.imread(test_dataset_vis.images_fps[i])
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    orig_image = cv2.resize(orig_image, (480, 480))

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    pr_mask = np.repeat(pr_mask[:, np.newaxis, :, :], 3, axis=1) * converted_color_map
    pr_mask = np.sum(pr_mask, axis=0).transpose((1, 2, 0)).astype(np.uint8)
    orig_image += pr_mask
    # get each image chunk --> put to another image --> split if 2 lines --> OCR for each chunk

    orig_image = cv2.resize(orig_image, (500, 300))
    # image = image.astype(np.uint8).transpose((1, 2, 0)) + pr_mask

    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Test", orig_image)
    key = cv2.waitKey(-1)

    if key == ord('q'):
        break
