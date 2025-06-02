import tensorflow as tf
from tensorflow.keras.applications import ResNet50# type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
import numpy as np
from preprocess_data import train_model_2

x, y= train_model_2()

x_rgb= np.repeat(x, 3, axis= 3)
img_size= 64

x_train, x_val , y_train, y_val= train_test_split(x_rgb, y, test_size= 0.2, random_state= 42)
num_class= len(np.unique(y))

y_train_cat= to_categorical(y_train, num_class)
y_val_cat= to_categorical(y_val, num_class)

model= ResNet50(input_shape= (img_size, img_size, 3), include_top= False, weights= 'imagenet')

model.trainable= False

for layer in model.layers[:-30]:
    layer.trainable= False

model= Sequential([
    model,
    GlobalAveragePooling2D(),  
    BatchNormalization(),
    Dense(100, activation= 'relu'),
    Dropout(0.2),
    Dense(num_class, activation= 'softmax')
])

early_stop= EarlyStopping(monitor= 'val_loss', patience= 5, restore_best_weights= True)
model.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics= ['accuracy'])
model.summary()

model.fit(x_train, y_train_cat, batch_size= 20, validation_data= (x_val, y_val_cat), epochs= 200)