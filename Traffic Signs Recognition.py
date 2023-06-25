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
path_dataset = path_dataset + '\dataset'
#train_dataset_str = path_dataset + '\Train'
train_dataset = os.listdir(path_dataset)
# testDir_list = os.listdir(path_dataset + '\Test')

classes = {
    0:'Roundabout mandatory',
    1: 'Stop',
    2: 'Turn left ahead',
    3: 'Ahead only',
    4: 'Green Light',
    5: 'Red Light',
    }


images = []
images_labels = []

for i in range(6):  # len(train_dataset)
    path = path_dataset + '\\' + str(i)
    train_img = os.listdir(path)

    for img in train_img:
        image = Image.open(path + '\\' + img)
        image = image.resize((30, 30))
        image = np.array(image)
        images.append(image)
        images_labels.append(i)

images = np.array(images)
images_labels = np.array(images_labels)

print(images.shape, images_labels.shape)

X_train, X_val, y_train, y_val = train_test_split(images, images_labels, test_size=0.2, random_state=42, shuffle=True)


print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
print(X_train)
y_train = tensorflow.keras.utils.to_categorical(y_train, 6)
y_val = tensorflow.keras.utils.to_categorical(y_val, 6)

model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                                   input_shape=X_train.shape[1:]),
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

    tensorflow.keras.layers.Dense(6, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


epochs = 10
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_val, y_val))


model.save("model3.h5")
