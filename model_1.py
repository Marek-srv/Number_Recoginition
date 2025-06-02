import tensorflow as tf
from tensorflow.keras import layers, models #type: ignore

def create_model(input_shape, num_classes):
    model= models.Sequential([
        layers.Conv2D(32, (3, 3), activation= 'relu', input_shape= input_shape,),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(32, (3 ,3), activation= 'relu', kernel_initializer='he_normal'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(50, activation= 'relu', kernel_initializer='he_normal'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation= 'softmax')
    ])
    optimizer= tf.keras.optimizers.Adam(learning_rate= 0.008)
    model.compile(optimizer= optimizer, loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])
    model.summary()
    return model