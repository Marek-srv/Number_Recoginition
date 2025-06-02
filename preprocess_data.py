import cv2
import os
import numpy as np
import random
from collections import Counter


data_dir= "/Users/sriramvishals/Srv_Project_digital_Rec/data"
img_size= 64

x=[]
y=[]

def augment_image(img):
    angle= random.uniform(-50,50)
    img_s= int(img_size/2)
    M= cv2.getRotationMatrix2D((img_s, img_s), angle, 1)
    rotated= cv2.warpAffine(img, M, (img_size, img_size))

    if random.choice([True, False]):
        rotated= cv2.flip(rotated, 1)
    return rotated


for digit in os.listdir(data_dir):
    digit_dir= os.path.join(data_dir, str(digit))

    if not os.path.isdir(digit_dir):
        continue

    for filename in os.listdir(digit_dir):
        img_path= os.path.join(digit_dir, filename)
        img= cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"couldn't read skipping")
            continue
        img= cv2.resize(img, (img_size, img_size))
        img= img/ 255.0
        x.append(img)
        y.append(int(digit))


        augmented_img= augment_image(img)
        augmented_img= augmented_img/ 255.0
        x.append(augmented_img)
        y.append(int(digit))


print(f"Length of X: {len(x)}")
print(f"Length of Y: {len(y)}")

x= np.array(x).reshape(-1, img_size, img_size, 1)
y= np.array(y)

def train_model_2():
    return x, y

print(f"Data Shape: {x.shape}")
print(f"\nLabels shape: {y.shape}")

np.save("x.npy", x)
np.save('y.npy', y)

print(f"Digit count distribution:", Counter(y))

print("\n Preprocessing is done")
