import cozmo
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

classes = {
    0:'Roundabout mandatory',
    1: 'Stop',
    2: 'Turn left ahead',
    3: 'Ahead only',
    4: 'Green Light',
    5: 'Red Light',
    }


def nothing(x):
    pass

cv.namedWindow("Trackbars")
cv.createTrackbar("HeightLower", "Trackbars", 46, 240, nothing)
cv.createTrackbar("WidthLower", "Trackbars", 176, 300, nothing)
cv.createTrackbar("HeightUpper", "Trackbars", 145, 240, nothing)
cv.createTrackbar("WidthUpper", "Trackbars", 281, 320, nothing)
def verifying_rgb_image(rgb_image):
    rgb_image = np.array(rgb_image)

    if np.all(rgb_image[:, :, 0] == rgb_image[:, :, 1]) and np.all(rgb_image[:, :, 0] == rgb_image[:, :, 2]):
        return 0
    else:
        return 1

def region_of_interest(image):
    height, width, _ = image.shape



    height_t_l = cv.getTrackbarPos("HeightLower", "Trackbars")
    width_t_l = cv.getTrackbarPos("WidthLower", "Trackbars")
    height_t_u = cv.getTrackbarPos("HeightUpper", "Trackbars")
    width_t_u = cv.getTrackbarPos("WidthUpper", "Trackbars")
    bgr_image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

    bgr_image = bgr_image[height_t_l:height_t_u,width_t_l:width_t_u,:]
    #bgr_image = image[65:167, 176:300, :]
    return bgr_image


def line_follower(robot: cozmo.robot.Robot):
    battery_voltage = robot.battery_voltage
    print(battery_voltage)
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
    model = load_model("model2.h5")

    while True:
        latest_image = robot.world.latest_image
        if latest_image:
            latest_image_array = np.array(latest_image.raw_image)
            if verifying_rgb_image(latest_image_array):
                imag = region_of_interest(np.array(latest_image.raw_image))
                cv.imshow('Image processed for test',imag)
                rgb_image = cv.cvtColor(np.array(imag), cv.COLOR_BGR2RGB)
                imag = cv.resize(rgb_image, (30, 30))

                data = []
                data.append(np.array(imag))
                images = np.array(data)

                pred = model.predict(images)
                classes_x = np.argmax(pred, axis=1)
                pred = classes[classes_x[0]]


                print(pred)


                #rgb_image = cv.cvtColor(np.array(latest_image.raw_image), cv.COLOR_BGR2RGB)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
# imag = cv.imread('stop.png')

    # print(imag.shape)



# try:
cozmo.run_program(line_follower, use_viewer=True)