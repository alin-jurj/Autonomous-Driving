import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv

classes = {
    0:'Roundabout mandatory',
    1: 'Stop',
    2: 'Turn left ahead',
    3: 'Ahead only',
    4: 'Green Light',
    5: 'Red Light',
    }

def verifying_rgb_image(rgb_image):
    rgb_image = np.array(rgb_image)
    if np.all(rgb_image[:, :, 0] == rgb_image[:, :, 1]) and np.all(rgb_image[:, :, 0] == rgb_image[:, :, 2]):
        return 0
    else:
        return 1

def region_of_interest(image):
    height, width, _ = image.shape
    bgr_image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    image = bgr_image[46:145,176:281,:]
    return image

def recognition_and_drive(robot: cozmo.robot.Robot):
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True

    model = load_model("model2.h5")

    robot.set_lift_height(0.0).wait_for_completed()
    robot.drive_straight(distance_mm(270), speed_mmps(50)).wait_for_completed()
    robot.turn_in_place(degrees(-10)).wait_for_completed()

    while True:
        robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        while True:
            latest_image = robot.world.latest_image
            if latest_image:
                latest_image_array = np.array(latest_image.raw_image)
                if verifying_rgb_image(latest_image_array):
                    break

        imag = region_of_interest(np.array(latest_image_array))

        rgb_image = cv.cvtColor(np.array(imag), cv.COLOR_BGR2RGB)
        imag = cv.resize(rgb_image, (30, 30))
        #

        data = []
        data.append(np.array(imag))
        images = np.array(data)

        pred = model.predict(images)
        classes_x = np.argmax(pred, axis=1)

        print(classes_x[0])
        if classes_x[0] == 1:
            robot.say_text("STOP STOP").wait_for_completed()
            break

        if classes_x[0] == 5:
            robot.say_text("Red Light").wait_for_completed()

        if classes_x[0] == 4:
            robot.say_text("Green Light").wait_for_completed()
            break


    robot.turn_in_place(degrees(13)).wait_for_completed()
    robot.drive_straight(distance_mm(200), speed_mmps(50)).wait_for_completed()