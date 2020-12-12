import cv2
import os
import glob

WINDOW_NAME = "example"
rectangles = []
pivots = []

# identity_label_positions = [
#     [(234, 71), (445, 99)],  # identity number
#     [(147, 109), (480, 158)],  # name
#     [(150, 163), (415, 187)],  # birthday
#     [(153, 192), (484, 244)],  # countryside
#     [(155, 246), (481, 298)]  # current address
# ]

identity_label_positions = [[(240, 68), (451, 102)], [(150, 96), (496, 159)], [(151, 158), (458, 188)], [(151, 188), (498, 243)], [(154, 240), (498, 297)]]


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


if __name__ == "__main__":
    img_paths = glob.glob("dataset/train2/*.jpg")
    # img_paths = glob.glob("dataset/train2/130307241_3454203954699208_4550129914180224305_o.jpg")

    for path in img_paths:
        print(path)

        img = cv2.imread(path)
        img = append_positions(img, positions=identity_label_positions)
        key = show_image(img)
        print(rectangles)

        if key == ord("q"):
            break
