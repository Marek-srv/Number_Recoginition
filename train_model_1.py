import numpy as np
from sklearn.model_selection import train_test_split
from model import create_model
import matplotlib.pyplot as plt

x= np.load('x.npy')
y= np.load('y.npy')

input_shape= (64, 64, 1)
num_classes= len(np.unique(y))

model= create_model(input_shape, num_classes)
model.summary()

history= model.fit(x, y, batch_size=20, epochs= 200, validation_split= 0.2)

loss, acc= model.evaluate(x, y)
print(f" Accuracy: {acc}")

model.save("Model_1.keras")
print("Model_1 saved successfully")

"""
model.save("number_recogintion.keras")
print("Saved")





epochs= list(range(1, len(history.history['accuracy'])+1))


plt.plot(epochs, history.history['accuracy'])
plt.figure(figsize= (13, 7))
plt.xticks(epochs)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

"""