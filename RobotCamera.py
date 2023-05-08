import cozmo
import cv2 as cv
from queue import Queue
import matplotlib.pylab as plt
import numpy as np
import threading
import time
import math
from cozmo.util import degrees, distance_mm, speed_mmps

CameraImages = Queue()
Steering_Angles = Queue()


def make_points(image, average):
    height, width = image.shape
    slope, intercept = average

    y1 = height
    y2 = int(y1 * 1 / 2)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [x1, y1, x2, y2]


def make_half_image(image):
    height, width = image.shape

    mask = np.zeros_like(image)

    polygon = np.array([[
        (0, height * 1 / 2), (width, height * 1 / 2),
        (width, height), (0, height), ]], np.int32)
    cv.fillPoly(mask, polygon, 255)
    filtered_image = cv.bitwise_and(image, mask)

    return filtered_image


def process_image():
    while True:
        image = CameraImages.get()
        while image is None:
            image = CameraImages.get()
            if image:
                break
        pil_image = image.raw_image
        #gray_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2GRAY)
        gray_image = np.array(pil_image)

        canny_image = cv.bitwise_not(gray_image)
        canny_image = cv.Canny(gray_image, 180, 255)

        take_half_image = make_half_image(canny_image)
        lines = cv.HoughLinesP(take_half_image, 1, np.pi / 100, 10, minLineLength=10, maxLineGap=4)

        left = []
        right = []

        height, width = canny_image.shape
        if len(lines) == 0:
            steering_angle = 0
        else:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)

                if x1 == x2:
                    continue
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                y_int = parameters[1]

                if slope < 0:
                    if x1 < width / 2 and x2 < width / 2:
                        left.append((slope, y_int))
                else:
                    if x1 > width / 2 and x2 > width / 2:
                        right.append((slope, y_int))
            lanes = []

            if len(left) > 0:
                left_avg = np.average(left, axis=0)
                lanes.append(make_points(canny_image, left_avg))

            if len(right) > 0:
                right_avg = np.average(right, axis=0)
                lanes.append(make_points(canny_image, right_avg))


            if len(lanes) > 1:
                _, _, left_x2, _ = lanes[0]
                _, _, right_x2, _ = lanes[1]
                x_offset = (left_x2 + right_x2) / 2 - width / 2
                y_offset = int(height / 2)

                angle_radian = math.atan(x_offset / y_offset)
                angle_degree = int(angle_radian * 180.0 / math.pi)
                steering_angle = angle_degree % 90
            else:
                if len(lanes) == 1:
                    x1, _, x2, _ = lanes[0]
                    x_offset = x2 - x1
                    y_offset = int(height/2)

                    angle_radian = math.atan(x_offset / y_offset)
                    angle_degree = int(angle_radian * 180.0 / math.pi)
                    steering_angle = angle_degree % 90
                else:
                    steering_angle = 0

        if 20 > steering_angle > 0:
            steering_angle = steering_angle - 20

        if Steering_Angles.full:
            with Steering_Angles.mutex:
                Steering_Angles.queue.clear()
        Steering_Angles.put(steering_angle)
        # print(Steering_Angles.qsize())


def drive(robot: cozmo.robot.Robot = None):
     robot.set_head_angle(cozmo.robot.MIN_HEAD_ANGLE,
                          in_parallel=True).wait_for_completed()
     #robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
     while True:
        # robot.say_text("Andrei is going home").wait_for_completed()
        robot.drive_straight(distance_mm(150), speed_mmps(40)).wait_for_completed()

        steering = Steering_Angles.get()
        print(steering)
        while steering is None:
            steering = Steering_Angles.get()
            if steering:
                break

        robot.turn_in_place(degrees(steering)).wait_for_completed()

     return


def RobotCamera(robot: cozmo.robot.Robot = None):
    # robot.say_text("Andrei is going home").wait_for_completed()
    robot.camera.image_stream_enabled = True

    while True:
        latest_image = robot.world.latest_image
        if latest_image:
            break

    while latest_image:
        if CameraImages.full:
            with CameraImages.mutex:
                CameraImages.queue.clear()
        CameraImages.put(latest_image)
        # gray_image = cv.cvtColor(latest_image, cv.COLOR_RGB2GRAY)
        # print(CameraImages.qsize())
        while True:
            latest_image = robot.world.latest_image
            if latest_image:
                break

    robot.camera.image_stream_enabled = False


def line_follower(robot: cozmo.robot.Robot):
    # CameraThread = threading.Thread(target=RobotCamera, args=(robot,))
    Process_image_Thread = threading.Thread(target=process_image)
    driveThread = threading.Thread(target=drive, args=(robot,))
    # CameraThread.start()
    Process_image_Thread.start()
    driveThread.start()
    RobotCamera(robot)

    # image = cv.imread('curba_ok.png')
    # steering_value = process_image(image)
    # print(steering_value)
    # plt.imshow(image)
    # plt.show()


try:
    cozmo.run_program(line_follower)
    # cozmo.run_program(run, use_viewer=True, force_viewer_on_top=True)
except SystemExit as e:
    print('exception = "%s"' % e)
    # ONLY FOR TESTING PURPOSES
    print('\nGoing on without Cozmo: for testing purposes only!', 'red')
    line_follower(None)
