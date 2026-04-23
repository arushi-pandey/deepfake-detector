import os
import cv2
import numpy as np

IMG_SIZE = 128

def load_data(data_dir):
    data = []
    labels = []

    for category in ["real", "fake"]:
        path = os.path.join(data_dir, category)
        label = 0 if category == "real" else 1

        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                data.append(image)
                labels.append(label)
            except:
                pass

    return np.array(data)/255.0, np.array(labels)