
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import tensorflow

from keras import layers
# from tensorflow import keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam


IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3

path_dataset = os.getcwd()
path_dataset = path_dataset + '\Dataset'

testDir_list = os.listdir(path_dataset)

categories = {
    0: 'Cedeaza trecerea',
    1: 'Stop',
    2: 'Accesul interzis'
}

print(path_dataset)
print(testDir_list)


images = []
images_labels = []

for i in range(len(testDir_list)):
    path = path_dataset + '\\' + str(i)
    train_img = os.listdir(path)

    for img in train_img:
        image = cv2.imread(path+ '\\' + img)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_array = Image.fromarray(image, 'RGB')
        image_array = image_array.resize((30, 30))
        images.append(np.array(image_array))
        images_labels.append(i)

images = np.array(images)
images_labels = np.array(images_labels)

print(images.shape, images_labels.shape)

shuffle_indexes = np.arange(images.shape[0])
np.random.shuffle(shuffle_indexes)
images = images[shuffle_indexes]
image_labels = images_labels[shuffle_indexes]

X_train, X_val, y_train, y_val = train_test_split(images, images_labels, test_size=0.3, random_state=42, shuffle=True)

X_train = X_train/255
X_val = X_val/255



y_train = tensorflow.keras.utils.to_categorical(y_train, len(testDir_list))
y_val = tensorflow.keras.utils.to_categorical(y_val, len(testDir_list))

model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
                        input_shape=(IMG_HEIGHT,IMG_WIDTH,channels)),
    tensorflow.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tensorflow.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tensorflow.keras.layers.BatchNormalization(axis=-1),

    tensorflow.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tensorflow.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    tensorflow.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tensorflow.keras.layers.BatchNormalization(axis=-1),

    tensorflow.keras.layers.Flatten(),
    tensorflow.keras.layers.Dense(512, activation='relu'),
    tensorflow.keras.layers.BatchNormalization(),
    tensorflow.keras.layers.Dropout(rate=0.5),

    tensorflow.keras.layers.Dense(3, activation='softmax')
])

lr = 0.001
epochs = 10
opt = tensorflow.keras.optimizers.Adam(learning_rate=lr, weight_decay=lr / (epochs * 0.5))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
aug = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

history = model.fit(aug.flow(X_train, y_train, batch_size=32), epochs=epochs, validation_data=(X_val, y_val))

model.save("model.h5")
