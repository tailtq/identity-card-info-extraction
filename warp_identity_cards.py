import os

import torch
import cv2
import numpy as np
import shutil

from utils.common import predict
from utils.datasets import letterbox
import glob
import imutils


def perspective_transform(image, source_points):
    dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (500, 300))

    return dst


def convert_img(img, device, half, new_size=416):
    img = letterbox(img, new_shape=new_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)

    img = img.half() if half else img.float()
    img = img / 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img


def get_center(coordinates):
    tl_x, tl_y = coordinates[:2]
    br_x, br_y = coordinates[2:4]

    return round((tl_x + br_x) / 2), round((tl_y + br_y) / 2)


def filter_redundancy(result):
    deleted_indexes = []

    for i, element in enumerate(result):
        category = element[5]
        max_class_confidence_score = max(result[np.where(result[:, 5] == category)][:, 4])

        if element[4] < max_class_confidence_score:
            deleted_indexes.append(i)

    return np.delete(result, deleted_indexes, axis=0)


if __name__ == "__main__":
    img_paths = glob.glob("dataset/new/*.jpg") + glob.glob("dataset/train/*.jpg") + glob.glob("dataset/val/*.jpg")

    # move new images
    # img_paths = list(filter(lambda img_path: not os.path.exists(img_path.replace(".jpg", ".txt")), img_paths))
    # for path in img_paths:
    #     shutil.move(path, "dataset/new/")
    # exit()

    for img_path in img_paths:
        img_name = img_path.split('/')[-1]
        result, orig_img, plot_img = predict(img_path)

        # filter redundant points by comparing confidence score
        if len(result) > 4:
            result = filter_redundancy(result)

        if len(result) == 4:
            coordinates = np.float32([get_center(result[index]) for index in np.argsort(result[:, 5])])
            orig_img = perspective_transform(orig_img, coordinates)

            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"dataset/output/{img_name}", orig_img)
        else:
            print(f"Failed: {img_path}, number of coordinates: {len(result if result is not None else [])}")
            print('---------------------------------------------------------------------')

        # plot_img = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
        # plot_img = imutils.resize(plot_img, height=500)
        # cv2.imshow("Test", plot_img)
        # key = cv2.waitKey(-1)
        #
        # if key == ord('q'):
        #     break
