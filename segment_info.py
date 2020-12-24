import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

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

# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CATEGORIES,
)
color_map = get_color_map()[1:]
converted_color_map = np.repeat(np.repeat(color_map[:, :, np.newaxis, np.newaxis], 480, axis=2), 480, axis=3)

for image, gt_mask in test_dataset_vis:
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    pr_mask = np.repeat(pr_mask[:, np.newaxis, :, :], 3, axis=1) * converted_color_map
    pr_mask = np.sum(pr_mask, axis=0).transpose((1, 2, 0)).astype(np.uint8)

    pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_RGB2BGR)
    cv2.imshow("Test", pr_mask)
    key = cv2.waitKey(-1)

    if key == ord('q'):
        break
