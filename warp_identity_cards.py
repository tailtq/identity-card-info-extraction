import cv2
import glob
import imutils

from utils.object_recognition_common import load_model, warp_identity_card

model = load_model("warping_model.pt")

if __name__ == "__main__":
    img_paths = glob.glob("dataset/4_angles_train/*.jpg") + glob.glob("dataset/4_angles_val/*.jpg")

    for img_path in img_paths:
        img_name = img_path.split('/')[-1]
        img = cv2.imread(img_path)

        try:
            img = warp_identity_card(img, model)
            cv2.imwrite(f"dataset/output/{img_name}", img)

            img = imutils.resize(img, height=500)
            cv2.imshow("Test", img)
            key = cv2.waitKey(-1)
            if key == ord('q'):
                break
        except ValueError:
            continue
