import torch
import cv2
import sys
import imutils
import numpy as np

from utils.datasets import letterbox
from utils.general import non_max_suppression

sys.path.append('')

from models.experimental import attempt_load

MODEL_PATH = "test/identity-card-alignment/best.pt"
CLASSES = ["top_left", "top_right", "bottom_right", "bottom_left"]
device = torch.device("cuda:0")


def load_model(path):
    return attempt_load(path, map_location=device)


def perspective_transform(image, source_points):
    print(source_points, image.shape)
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


def draw_bbox(coordinates, img, classes):
    for coordinate in coordinates:
        cv2.rectangle(img, tuple(coordinate[0:2].tolist()), tuple(coordinate[2:4].tolist()), (0, 255, 0), 2)
        cv2.putText(img, classes[coordinate[5]], tuple(coordinate[0:2].tolist()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img


def get_center(top_left, bottom_right):
    tl_x, tl_y = top_left
    br_x, br_y = bottom_right

    return ((tl_x + br_x) // 2).type(torch.int).item(), \
           ((tl_y + br_y) // 2).type(torch.int).item()


if __name__ == '__main__':
    half = device.type != 'cpu'
    model = load_model(MODEL_PATH)

    if half:
        model.half()

    img = cv2.imread("test/identity-card-alignment/Net_Phan Quoc Viet.JPG")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_img = imutils.resize(img, width=1080)
    orig_height, orig_width, _ = orig_img.shape

    img = convert_img(orig_img, device, half, new_size=480)
    _, _, new_height, new_width = img.size()

    preds = model(img)[0]
    preds = non_max_suppression(preds, 0.4, 0.5)

    for pred in preds:
        pred = pred.type(torch.int)
        pred[:, 0] = (pred[:, 0] * orig_width // new_width)
        pred[:, 1] = (pred[:, 1] * orig_height // new_height)
        pred[:, 2] = (pred[:, 2] * orig_width // new_width)
        pred[:, 3] = (pred[:, 3] * orig_height // new_height)
        orig_img = draw_bbox(pred, orig_img, CLASSES)

        coordinates = {
            CLASSES[pred[0, 5]]: get_center(pred[0, 0:2], pred[0, 2:4]),
            CLASSES[pred[1, 5]]: get_center(pred[1, 0:2], pred[1, 2:4]),
            CLASSES[pred[2, 5]]: get_center(pred[2, 0:2], pred[2, 2:4]),
            CLASSES[pred[3, 5]]: get_center(pred[3, 0:2], pred[3, 2:4]),
        }
        source_points = np.float32([
            coordinates["top_left"], coordinates["top_right"], coordinates["bottom_right"], coordinates["bottom_left"]
        ])
        # orig_img = perspective_transform(orig_img, source_points)

        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Show identity card", orig_img)
        cv2.waitKey(-1)

