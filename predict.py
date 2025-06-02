import cv2
import numpy as np
import tensorflow as tf

img= cv2.imread("", cv2.IMREAD_COLOR)
img= cv2.resize(img, (64, 64))
img= img/ 255.0
img= img.reshape(1, 64, 64, 3)


model= tf.keras.models.load_model('number_recogintion.keras')
pred= model.predict(img)

print(f"Predicted_Digit:{np.argmax(pred)}")