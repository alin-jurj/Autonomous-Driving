import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

classes = {
    0:'Roundabout mandatory',
    1: 'Stop',
    2: 'Turn left ahead',
    3: 'Ahead only',
    4: 'Green Light',
    5: 'Red Light',
    }


model = load_model("model2.h5")
imag = cv.imread('green_light_2.png')
rgb_image = cv.cvtColor(np.array(imag), cv.COLOR_BGR2RGB)
imag = cv.resize(rgb_image,(30,30))

plt.imshow(imag)
plt.show()
data = []
data.append(np.array(imag))
images = np.array(data)

pred = model.predict(images)
classes_x = np.argmax(pred, axis=1)
print(pred)
print(classes[classes_x[0]])