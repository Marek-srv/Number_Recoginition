import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # type: ignore
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

model= MobileNetV2(input_shape= (img_size, img_size, 3), include_top= False, weights= 'imagenet')

model.trainable= True

for layer in model.layers[:-20]:
    layer.trainable= False


model= Sequential([
    model,
    GlobalAveragePooling2D(),
    Dense(200, activation= 'relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(100, activation= 'relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_class, activation= 'softmax')
])

early_stop= EarlyStopping(monitor= 'val_loss', patience= 5, restore_best_weights=  True)
model.compile(optimizer= Adam(learning_rate= 0.005), loss= 'categorical_crossentropy', metrics= ['accuracy'] )
model.summary()

img_change= ImageDataGenerator(
    rotation_range= 20,
    zoom_range= 0.15,
    width_shift_range= 0.2,
    height_shift_range= 0.2,
    horizontal_flip= True
)

img_change.fit(x_train)
model.fit(x_train, y_train_cat, batch_size= 20, validation_data= (x_val, y_val_cat), epochs= 200, callbacks= [early_stop])

model.save("Model_2.keras")
print("Model_2 saves Successfully")