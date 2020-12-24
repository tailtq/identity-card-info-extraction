import cv2
import imutils
import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, plot_one_box

device = torch.device("cuda:0")
half = device.type != "cpu"

CATEGORIES = ["top_left", "top_right", "bottom_right", "bottom_left"]
COLORS = [(66, 135, 245), (194, 66, 245), (250, 52, 72), (111, 250, 52)]


def load_model(path):
    model = attempt_load(path, map_location=device)

    if half:
        model.half()

    return model


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


def predict_4_corners(img_path, model, resized_width=1080):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if resized_width is not None:
        orig_img = imutils.resize(img, width=1080)
    else:
        orig_img = img.copy()

    plot_img = orig_img.copy()
    img = convert_img(orig_img, device, half, new_size=480)
    _, _, new_height, new_width = img.size()

    preds = model(img)[0]
    preds = non_max_suppression(preds, 0.4, 0.5)
    result = np.array([], dtype=np.float32)

    # based on YOLOv5
    for i, det in enumerate(preds):  # detections per image
        if det is not None:
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.size()[2:], det[:, :4], orig_img.shape).round()
            result = det.type(torch.float32).cpu().detach().numpy()

            # for visualization only
            for *xyxy, conf, cls in reversed(det):
                label = '%s %.2f' % (CATEGORIES[int(cls)], conf)
                plot_one_box(xyxy, plot_img, label=label, color=COLORS[int(cls)], line_thickness=3)

    return result, orig_img, plot_img


def draw_bbox(coordinates, img, classes):
    for coordinate in coordinates:
        cv2.rectangle(img, tuple(coordinate[0:2].tolist()), tuple(coordinate[2:4].tolist()), (0, 255, 0), 2)
        cv2.putText(img, classes[coordinate[5]], tuple(coordinate[0:2].tolist()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img
