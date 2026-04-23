import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 128

model = load_model("model/deepfake_model.h5")

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img)/255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        print("Fake Image")
    else:
        print("Real Image")

predict_image("test.jpg")