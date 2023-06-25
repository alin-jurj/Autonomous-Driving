import sys

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
cv.createTrackbar("L-H","Trackbars",0,179, nothing)
cv.createTrackbar("L-S","Trackbars",26,255, nothing)
cv.createTrackbar("L-V","Trackbars",68,255, nothing)

cv.createTrackbar("U-H","Trackbars",102,179, nothing)
cv.createTrackbar("U-S","Trackbars",170,255, nothing)
cv.createTrackbar("U-V","Trackbars",210,255, nothing)

def verifying_rgb_image(rgb_image):
    rgb_image = np.array(rgb_image)
    if np.all(rgb_image[:, :, 0] == rgb_image[:, :, 1]) and np.all(rgb_image[:, :, 0] == rgb_image[:, :, 2]):
        return 0
    else:
        return 1

def region_of_interest(image):
    height, width, _ = image.shape
    image = image[:,width-100:,:]
    return image
def make_half_image(image):
    height, width = image.shape

    mask = np.zeros_like(image)

    polygon = np.array([[
        (0, height * 1 / 2), (width, height * 1 / 2),
        (width, height), (0, height), ]], np.int32)
    cv.fillPoly(mask, polygon, 255)
    filtered_image = cv.bitwise_and(image, mask)

    return filtered_image
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

                #imag = region_of_interest(rgb_image)

                #cv.imshow("Prediction",imag)
                #plt.show()
                #green = green_mark(rgb_image)
                green = green_frame(rgb_image)

               # green_e= cv.erode(green, (3,3), iterations = 2)
                res = cv.bitwise_and(rgb_image, rgb_image, mask = green)

                canny_image = cv.Canny(res, 100, 255)
                #canny_image = cv.dilate(canny_image, (3, 3), iterations=1)
                #canny_image = cv.adaptiveThreshold(canny_image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
                #canny_image = make_half_image(canny_image)
                # #change image perspective
                # height, width = canny_image.shape
                # tl = (height/2,0)
                # bl = (height,0)
                # tr = (height/2, width)
                # br = (height,width)
                #
                # pts1 = np.float32([tl, bl, tr, br])
                # pts2 = np.float32([[0,0], [0,width], [height,0], [height,width]])
                # matrix = cv.getPerspectiveTransform(pts1,pts2)
                # perspective_transformed = cv.warpPerspective(canny_image,matrix, (height, width))
                cv.imshow('Canny',canny_image)
                #cv.imshow("Fara erodare", green)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                #plt.show()
                # plt.pause(0.5)

                break
            latest_image = robot.world.latest_image

    cv.destroyAllWindows()


cozmo.run_program(cozmo_program, use_viewer=True)