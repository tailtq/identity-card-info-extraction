import os
import glob
import cv2
import json
import torch
import numpy as np
import albumentations as albu
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset as BaseDataset

CATEGORIES_WITH_ID = [
    {
        "id": 0,
        "name": "background",
    },
    {
        "id": 1,
        "name": "name",
    },
    {
        "id": 2,
        "name": "birthday",
    },
    {
        "id": 3,
        "name": "countryside",
    },
    {
        "id": 4,
        "name": "address",
    },
    {
        "id": 5,
        "name": "identity number",
    },
]
CATEGORIES = list(map(lambda e: e["name"], CATEGORIES_WITH_ID))


def load_model():
    return torch.load('./info_segmentation.pth')


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """
    CLASSES = ['background', 'address', 'birthday', 'countryside', 'identity number', 'name']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
            image_links=[]
    ):
        if len(image_links):
            self.images_fps = image_links
            self.masks_fps = []
        else:
            self.images_fps = glob.glob(f"{images_dir}/*.jpg")
            self.masks_fps = glob.glob(f"{masks_dir}/*.png")

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (480, 480))

        # for test dataset
        if len(self.masks_fps) - 1 < i:
            mask = np.zeros(image.shape, dtype=np.float)
        else:
            mask = cv2.imread(self.masks_fps[i], 0)

        mask = cv2.resize(mask, (480, 480))
        #         extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images_fps)


class CocoDataset(Dataset):
    def __init__(self, coco_data, src_dir, classes=None, augmentation=None, preprocessing=None, from_id=0):
        image_links = []

        for image in coco_data["images"]:
            if image["id"] >= from_id:
                image_links.append(f"{src_dir}/{image['file_name']}")

        super().__init__("", "", classes, augmentation, preprocessing, image_links)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        #         albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    if len(x.shape) == 3:
        return x.transpose(2, 0, 1).astype('float32')

    return x.astype('float32')


def get_preprocessing_fn():
    ENCODER = 'vgg13_bn'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    return preprocessing_fn


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def write_annotation_file(content, file_name):
    json_annotations = json.dumps(content)
    writer = open(file_name, "w+")
    writer.write(json_annotations)
    writer.close()


def get_color_map(with_label=False):
    data = open("labelmap.txt", "r").read()
    lines = data.split("\n")
    lines.pop(0)
    lines.pop(len(lines) - 1)

    color_map = []

    for line in lines:
        color_map.append([int(color) for color in line.split(":")[1].split(",")])

    color_map = np.array(color_map)

    if with_label:
        return color_map, CATEGORIES_WITH_ID

    return color_map


# yx
def get_min_max_x_y(coordinates):
    min_y, min_x, max_y, max_x = min(coordinates[:, 0]), min(coordinates[:, 1]), max(coordinates[:, 0]), \
                                 max(coordinates[:, 1])

    return min_x, min_y, max_x, max_y


def get_segment(min_x_a, min_y_a, max_x_a, max_y_a, min_x_b=None, min_y_b=None, max_x_b=None, max_y_b=None):
    if min_x_b is not None:
        segment = np.array([
            [min_x_a, min_y_a],
            [max_x_a, min_y_a],
            [max_x_a, max_y_a],
            [max_x_b, min_y_b],
            [max_x_b, max_y_b],
            [min_x_b, max_y_b],
            [min_x_b, min_y_b],
            [min_x_a, max_y_a],
        ])
    else:
        segment = np.array([
            [min_x_a, min_y_a],
            [max_x_a, min_y_a],
            [max_x_a, max_y_a],
            [min_x_a, max_y_a],
        ])

    return segment
