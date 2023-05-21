import cozmo
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# def cozmo_program(robot: cozmo.robot.Robot):
#     # Abilitați camera robotului
#     robot.camera.image_stream_enabled = True
#     robot.camera.color_image_enabled = True
#     # print(robot.camera._color_image_enabled)
#     # # Așteptați până când se primește o imagine
#     # image = None
#     # while image is None:
#     #     event = robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage)
#     #     image = event.image
#     #     image = np.asarray(image)
#     #
#     # print(image.shape)
#     # # Afișați imaginea
#     # plt.imshow(image)
#     # plt.show()
#     # #cv.waitKey(0)
#     # #cv.destroyAllWindows()
#     while True:
#         pass
#
# cozmo.run_program(cozmo_program, use_viewer= True)
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    pass
cv.namedWindow("Trackbars")
cv.createTrackbar("L-H","Trackbars",26,179, nothing)
cv.createTrackbar("L-S","Trackbars",121,255, nothing)
cv.createTrackbar("L-V","Trackbars",66,255, nothing)

cv.createTrackbar("U-H","Trackbars",102,179, nothing)
cv.createTrackbar("U-S","Trackbars",153,255, nothing)
cv.createTrackbar("U-V","Trackbars",247,255, nothing)

def verifying_rgb_image(rgb_image):
    rgb_image = np.array(rgb_image)
    if np.all(rgb_image[:, :, 0] == rgb_image[:, :, 1]) and np.all(rgb_image[:, :, 0] == rgb_image[:, :, 2]):
        return 0
    else:
        return 1
def green_frame(image):
    hsv = cv.cvtColor(np.array(image), cv.COLOR_RGB2HSV)


    l_h = cv.getTrackbarPos("L-H", "Trackbars")
    l_s = cv.getTrackbarPos("L-S", "Trackbars")
    l_v = cv.getTrackbarPos("L-V", "Trackbars")

    u_h = cv.getTrackbarPos("U-H", "Trackbars")
    u_s = cv.getTrackbarPos("U-S", "Trackbars")
    u_v = cv.getTrackbarPos("U-V", "Trackbars")
    lower_green = np.array([l_h, l_s, l_v])
    upper_green = np.array([u_h, u_s, u_v])
    # lower_green = np.array([26, 109, 66])
    # upper_green = np.array([105, 153, 247])

    green_mask = cv.inRange(hsv, lower_green, upper_green)
    return green_mask
def cozmo_program(robot: cozmo.robot.Robot):
    # Abilitați camera robotului
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.set_head_angle(cozmo.robot.MIN_HEAD_ANGLE,
                        in_parallel=True).wait_for_completed()
    #robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
    print(robot.battery_voltage)
    while True:
        # Așteptați până când se primește o imagine
        latest_image = robot.world.latest_image
        while latest_image is not None:

            # convertim imaginea in format numpy


            if(verifying_rgb_image(latest_image.raw_image)==1):
            # Afișăm imaginea
                rgb_image = cv.cvtColor(np.array(latest_image.raw_image), cv.COLOR_BGR2RGB)
                # plt.imshow(rgb_image)
                # plt.show()
                green = green_frame(rgb_image)
                res = cv.bitwise_and(rgb_image, rgb_image, mask = green)
                cv.imshow('Image',green)
                cv.imshow("Image2", res)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                #plt.show()
                # plt.pause(0.5)

                break
            latest_image = robot.world.latest_image

    cv.destroyAllWindows()


cozmo.run_program(cozmo_program, use_viewer=True)