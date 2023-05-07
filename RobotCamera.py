import cozmo
import cv2 as cv
from queue import Queue
import matplotlib.pylab as plt
import numpy as np
import threading
import time
import math
CameraImages = Queue()
Steering_Angles = Queue()

def make_points(image, average):
    height, width = image.shape
    slope, intercept = average

    y1 = height
    y2 = int(y1 * 1/2)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [x1, y1, x2, y2]

def make_half_image(image):
    height, width = image.shape

    mask = np.zeros_like(image)

    polygon = np.array([[
        (0, height * 1/2), (width, height * 1/2),
        (width, height), (0, height), ]], np.int32)
    cv.fillPoly(mask, polygon,255)
    filtered_image = cv.bitwise_and(image, mask)

    return filtered_image

def process_image(image):

    while CameraImages.empty() == False:
        pass
    image = CameraImages.get()

    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    canny_image = cv.Canny(gray_image, 180, 255)
    canny_image = cv.bitwise_not(canny_image)
    take_half_image = make_half_image(canny_image)
    lines = cv.HoughLinesP(take_half_image, 1, np.pi/100, 10, minLineLength=10, maxLineGap=4)


    left = []
    right = []

    height, width, _ = image.shape

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        if x1 == x2:
            continue
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]

        if slope < 0:
            if x1 < width/2 and x2 < width/2:
                left.append((slope, y_int))
        else:
            if x1 > width/2 and x2 > width/2:
                right.append((slope, y_int))
    lanes = []

    if len(left) > 0:
        left_avg = np.average(left, axis = 0)
        lanes.append(make_points(canny_image, left_avg))

    if len(right) > 0:
        right_avg = np.average(right, axis = 0)
        lanes.append(make_points(canny_image, right_avg))

    _, _, left_x2, _ = lanes[0]
    _, _, right_x2, _ = lanes[1]

    x_offset = (left_x2 + right_x2) / 2 - width / 2
    y_offset = int(height / 2)

    angle_radian = math.atan(x_offset / y_offset)
    angle_degree = int(angle_radian * 180.0 / math.pi)
    steering_angle = angle_degree # * 90

    Steering_Angles.put(steering_angle)
def drive():



    return


def RobotCamera(self, robot: cozmo.robot.Robot = None):
    robot.camera.image_stream_enabled = True

    latest_image = robot.world.latest_image
    while latest_image:
        if CameraImages.full() == True:
            with CameraImages.mutex:
                CameraImages.queue.clear()
        CameraImages.put(latest_image)
        latest_image = robot.world.latest_image


CameraThread = threading.Thread(target=RobotCamera)
driveThread = threading.Thread(target=drive)
CameraThread.start()
driveThread.start()
# image = cv.imread('curba_ok.png')
# steering_value = process_image(image)
# print(steering_value)
# plt.imshow(image)
# plt.show()

