import cozmo

import numpy as np
import cv2 as cv

def verifying_rgb_image(rgb_image):
    rgb_image = np.array(rgb_image)
    if np.all(rgb_image[:, :, 0] == rgb_image[:, :, 1]) and np.all(rgb_image[:, :, 0] == rgb_image[:, :, 2]):
        return 0
    else:
        return 1

def cozmo_program(robot: cozmo.robot.Robot):

    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True

    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
    print(robot.battery_voltage)

    while True:
        latest_image = robot.world.latest_image
        if latest_image:
            latest_image_array = np.array(latest_image.raw_image)
            if verifying_rgb_image(latest_image_array):
                break

    if latest_image:
        rgb_image = cv.cvtColor(np.array(latest_image.raw_image), cv.COLOR_BGR2RGB)

        cv.imwrite("./dataset/red_light/red_light26.png", rgb_image)

cozmo.run_program(cozmo_program)