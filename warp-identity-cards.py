import torch
import cv2
import sys
import imutils
import numpy as np

from utils.datasets import letterbox
from utils.general import scale_coords, non_max_suppression, plot_one_box
import glob

sys.path.append('')

from models.experimental import attempt_load

MODEL_PATH = "dataset/best.pt"
names = ["top_left", "top_right", "bottom_right", "bottom_left"]
colors = [(66, 135, 245), (194, 66, 245), (250, 52, 72), (111, 250, 52)]
device = torch.device("cuda:0")


def load_model(path):
    return attempt_load(path, map_location=device)


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


def draw_bbox(coordinates, img, classes):
    for coordinate in coordinates:
        cv2.rectangle(img, tuple(coordinate[0:2].tolist()), tuple(coordinate[2:4].tolist()), (0, 255, 0), 2)
        cv2.putText(img, classes[coordinate[5]], tuple(coordinate[0:2].tolist()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img


def get_center(coordinates):
    tl_x, tl_y = coordinates[:2]
    br_x, br_y = coordinates[2:4]

    return round((tl_x + br_x) / 2), round((tl_y + br_y) / 2)


if __name__ == "__main__":
    half = device.type != "cpu"
    model = load_model(MODEL_PATH)

    if half:
        model.half()

    img_paths = glob.glob("dataset/train/*.jpg")

    for img_path in img_paths:
        img_name = img_path.split('/')[-1]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_img = imutils.resize(img, width=1080)
        plot_img = orig_img.copy()
        orig_height, orig_width, _ = orig_img.shape

        img = convert_img(orig_img, device, half, new_size=480)
        _, _, new_height, new_width = img.size()

        preds = model(img)[0]
        preds = non_max_suppression(preds, 0.4, 0.5)
        result = None

        for i, det in enumerate(preds):  # detections per image
            gn = torch.tensor(orig_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.size()[2:], det[:, :4], orig_img.shape).round()
                result = det.type(torch.float32).cpu().detach().numpy()

                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, plot_img, label=label, color=colors[int(cls)], line_thickness=3)

        deleted_indexes = []

        if result is not None and len(result) > 4:
            for i, element in enumerate(result):
                category = element[5]
                max_class_confidence_score = max(result[np.where(result[:, 5] == category)][:, 4])

                if element[4] < max_class_confidence_score:
                    deleted_indexes.append(i)

            result = np.delete(result, deleted_indexes, axis=0)

        if result is not None and len(result) == 4:
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
