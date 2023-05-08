import os
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import tensorflow
import pandas as pd
from keras import layers

from tensorflow import keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam


IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3

path_dataset = os.getcwd()
path_dataset = path_dataset + '\Dataset'
train_dataset_str = path_dataset + '\Train'
train_dataset = os.listdir(train_dataset_str)
testDir_list = os.listdir(path_dataset + '\Test')

classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'}

images = []
images_labels = []

for i in range(3):  # len(train_dataset)
    path = train_dataset_str + '\\' + str(i+13)
    train_img = os.listdir(path)

    for img in train_img:
        image = Image.open(path + '\\' + img)
        image = image.resize((70, 70))
        image = np.array(image)
        images.append(image)
        images_labels.append(i)

images = np.array(images)
images_labels = np.array(images_labels)

print(images.shape, images_labels.shape)

X_train, X_val, y_train, y_val = train_test_split(images, images_labels, test_size=0.2, random_state=42, shuffle=True)


print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
print(X_train)
y_train = tensorflow.keras.utils.to_categorical(y_train, 3)
y_val = tensorflow.keras.utils.to_categorical(y_val, 3)

model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                                   input_shape=(70,70,3)), #X_train.shape[1:]),
    tensorflow.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tensorflow.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tensorflow.keras.layers.BatchNormalization(axis=-1),

    tensorflow.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tensorflow.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    tensorflow.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tensorflow.keras.layers.BatchNormalization(axis=-1),

    tensorflow.keras.layers.Flatten(),
    tensorflow.keras.layers.Dense(256, activation='relu'),
    tensorflow.keras.layers.BatchNormalization(),
    tensorflow.keras.layers.Dropout(rate=0.5),

    tensorflow.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


epochs = 4
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_val, y_val))


model.save("model3.h5")
