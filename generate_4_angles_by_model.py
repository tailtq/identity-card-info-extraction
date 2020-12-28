from utils.object_recognition_common import predict_4_corners, load_model, filter_redundancy
import glob
import os

if __name__ == "__main__":
    model = load_model("dataset/best-v1.3.pt")

    img_paths = glob.glob("dataset/train/*.jpg")
    img_paths = list(filter(lambda img_path: not os.path.exists(img_path.replace(".jpg", ".txt")), img_paths))

    for img_path in img_paths:
        img_name = img_path.split("/")[-1].split(".")[0]
        result, orig_img, _ = predict_4_corners(img_path, model, None)

        text = ""
        height, width, _ = orig_img.shape

        # filter redundant points by comparing confidence score
        if len(result) > 4:
            result = filter_redundancy(result)

        for det in result:
            label = [
                str(int(det[5])),
                str((det[0] + det[2]) / 2 / width),
                str((det[1] + det[3]) / 2 / height),
                str(abs(det[0] - det[2]) / width),
                str(abs(det[1] - det[3]) / height),
            ]
            text += " ".join(label) + "\n"

        # shutil.copy(img_path, "dataset/train2/")
        stream = open(f"dataset/new/{img_name}.txt", "w+")
        stream.write(text)
        stream.close()
