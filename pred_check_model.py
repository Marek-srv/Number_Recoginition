import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model #type: ignore


img_path= '/Users/sriramvishals/Srv_Project_digital_Rec/pred_check/check_number.jpg'
img= cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
img_1= cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_1= cv2.resize(img_1, (64, 64))
img_1= img_1/ 255.0
img_1= np.expand_dims(img_1, axis= 0)
img= cv2.resize(img, (64, 64))
img= img/255.0
img= np.expand_dims(img, axis= 0)
 #image have been transformed and ready for model insertion

model_1= load_model('Model_1.keras')
model_2= load_model('Model_2.keras')
model_3= load_model('Model_3.keras')

#y_pred= model_1.predict(img_1)
#print("Using Model_1 (BUILT MODEL)")
#print("Predicted Class:", np.argmax(y_pred))

y_pred= model_2.predict(img)
print("Using Model_2 (MOBILENETV2)")
print("Predicted Class:", np.argmax(y_pred))

y_pred= model_3.predict(img)
print("Using Model_3 (RESNET50)")
print("Predicted Class:", np.argmax(y_pred))