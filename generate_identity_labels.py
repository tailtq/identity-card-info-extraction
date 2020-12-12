import numpy as np
import cv2
import glob
import json

WINDOW_NAME = "example"
rectangles = []
pivots = []

categories = [
    {
        "id": 5,
        "name": "identity number",
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
    }
]
identity_label_positions = [
    [(261, 69), (451, 97)],  # identity number
    [(249, 100), (459, 130)],  # name
    [(307, 162), (409, 187)],  # birthday
    [(151, 188), (498, 243)],  # countryside
    [(154, 240), (498, 297)]  # current address
]
json_annotations = []


def set_draw_event(event, x, y, flags, param):
    global pivots

    if event == cv2.EVENT_LBUTTONDOWN:
        pivots.append((x, y))

        cv2.drawMarker(param["img"], (x, y), (0, 255, 0), markerSize=2)
        cv2.imshow(WINDOW_NAME, param["img"])
    if event == cv2.EVENT_LBUTTONUP and len(pivots) == 2:
        cv2.rectangle(param["img"], pivots[0], pivots[1], (0, 0, 255), 1)
        rectangles.append(pivots)
        pivots = []

        cv2.imshow(WINDOW_NAME, param["img"])


def append_positions(img, positions):
    for position in positions:
        cv2.rectangle(img, position[0], position[1], (0, 255, 0), 1)

    return img


def show_image(img):
    cv2.imshow(WINDOW_NAME, img)
    cv2.setMouseCallback(WINDOW_NAME, set_draw_event, param={"img": img})

    return cv2.waitKey(-1)


def get_bbox(positions):
    x1, y1 = positions[0]
    x2, y2 = positions[1]
    # tl tr br bl
    point_1 = (x1, y1)
    point_2 = (x2, y1)
    point_3 = (x2, y2)
    point_4 = (x1, y2)

    points = np.array([point_1, point_2, point_3, point_4], dtype=float)

    return points.reshape((8,)).tolist()


def get_bbox_info(positions):
    x1, y1 = positions[0]
    x2, y2 = positions[1]
    width, height = float(x2 - x1), float(y2 - y1)

    return float(x1), float(y1), width, height


def get_coco_format(id, img_id, category_id, positions):
    # calculate area
    bbox_info = get_bbox_info(positions)

    return {
        "id": id,
        "image_id": img_id,
        "category_id": category_id,
        "segmentation": [get_bbox(positions)],
        "area": bbox_info[2] * bbox_info[3],
        "bbox": bbox_info,
        "iscrowd": 0,
        "attributes": {
            "occluded": False
        }
    }


if __name__ == "__main__":
    img_paths = glob.glob("dataset/train/*.jpg")
    current_id = 1

    for img_index, path in enumerate(img_paths):
        for category_index, category in enumerate(categories):
            positions = identity_label_positions[category_index]

            json_annotations.append(get_coco_format(current_id, img_index + 1, category["id"], positions))

            current_id += 1

    json_annotations = json.dumps(json_annotations)
    writer = open("test.json", "w+")
    writer.write(json_annotations)
    writer.close()

    # img_paths = glob.glob("dataset/train2/65714583_2267212970262502_1702528472108236800_o.jpg")
    #
    # for path in img_paths:
    #     print(path)
    #
    #     img = cv2.imread(path)
    #     img = append_positions(img, positions=identity_label_positions)
    #     key = show_image(img)
    #     print(rectangles)
    #
    #     if key == ord("q"):
    #         break
