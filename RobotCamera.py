import cozmo
import cv2 as cv
from queue import Queue
import matplotlib.pylab as plt
import numpy as np
import threading
import time
import math
import sklearn
import joblib
from cozmo.util import degrees, distance_mm, speed_mmps
from tensorflow.keras.models import load_model
import db_connection
import traffic_signs

CameraImages = Queue()
Steering_Angles = Queue()


def verifying_rgb_image(rgb_image):
    rgb_image = np.array(rgb_image)
    if np.all(rgb_image[:, :, 0] == rgb_image[:, :, 1]) and np.all(rgb_image[:, :, 0] == rgb_image[:, :, 2]):
        return 0
    else:
        return 1



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
        #pil_image = image.raw_image
        #gray_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2GRAY)
        rgb_image = cv.cvtColor(np.array(image.raw_image), cv.COLOR_BGR2RGB)
        #rgb_image = np.array(image.raw_image)


        hsv = cv.cvtColor(rgb_image, cv.COLOR_RGB2HSV)


        # lower_green = np.array([0, 26, 68])
        # upper_green = np.array([94, 170, 210])

        lower_green = np.array([0, 26, 68])
        upper_green = np.array([94, 170, 210])

        green_mask = cv.inRange(hsv, lower_green, upper_green)


        res = cv.bitwise_and(rgb_image, rgb_image, mask = green_mask)


        #canny_image = cv.bitwise_not(shadow_free_image)
        canny_image = cv.Canny(res, 100, 255)
        #canny_image = cv.dilate(canny_image, (3, 3), iterations=1)
        canny_image = make_half_image(canny_image)
        lines = cv.HoughLinesP(canny_image, 1, np.pi / 100, 10, minLineLength=10, maxLineGap=4)

        left = []
        right = []

        height, width = canny_image.shape
        if lines is None or len(lines) == 0:
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
                steering_angle = angle_degree
            else:
                if len(lanes) == 1:
                    x1, _, x2, _ = lanes[0]
                    x_offset = x2 - x1
                    y_offset = int(height/2)

                    angle_radian = math.atan(x_offset / y_offset)
                    angle_degree = int(angle_radian * 180.0 / math.pi)
                    steering_angle = angle_degree
                else:
                    steering_angle = 0

        if 20 > abs(steering_angle) > 0:
            steering_angle = 0

        if Steering_Angles.full:
            with Steering_Angles.mutex:
                Steering_Angles.queue.clear()
        Steering_Angles.put(steering_angle)
        # print(Steering_Angles.qsize())


def drive(robot: cozmo.robot.Robot = None):
     robot.set_head_angle(cozmo.robot.MIN_HEAD_ANGLE,
                         in_parallel=True).wait_for_completed()
     #robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
     #action1 = robot.drive_straight(distance_mm(50), speed_mmps(25), should_play_anim=False, in_parallel=True)
     #action2 = None
     while True:
        # robot.say_text("Andrei is going home").wait_for_completed()
        steering = Steering_Angles.get()
        print(steering)
        while steering is None:
            steering = Steering_Angles.get()
            if steering:
                break

        if steering > 0:
            if steering > 60:
                robot.drive_wheels(50,10)
            else:
                if steering > 40:
                    robot.drive_wheels(40,10)
                else:
                    robot.drive_wheels(50,25)
            #robot.drive_wheels(50,25)
        if steering < 0:
            if steering < -60:
                robot.drive_wheels(10,50)
            else:
                if steering < -40:
                    robot.drive_wheels(10,40)
                else:
                    robot.drive_wheels(25,50)
        if steering == 0:
            robot.drive_wheels(18,18)

     return


def RobotCamera(robot: cozmo.robot.Robot = None):

    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    while True:
        latest_image = robot.world.latest_image
        if latest_image:
            latest_image_array = np.array(latest_image.raw_image)
            if verifying_rgb_image(latest_image_array):
                break

    while latest_image:
        if CameraImages.full:
            with CameraImages.mutex:
                CameraImages.queue.clear()
        CameraImages.put(latest_image)

        while True:
            latest_image = robot.world.latest_image
            if latest_image:
                latest_image_array = np.array(latest_image.raw_image)
                if verifying_rgb_image(latest_image_array):
                    break

    robot.camera.image_stream_enabled = False


def line_follower(robot: cozmo.robot.Robot):

    while True:
        ROAD = db_connection.db_location()
        if ROAD:
            break
        time.sleep(3)
    traffic_signs.recognition_and_drive(robot)

    ###### 1 - DRUMMMMMMMMMMMMMMMM
    CameraThread = threading.Thread(target=RobotCamera, args=(robot,))
    if ROAD == 1:
        robot.drive_straight(distance_mm(80), speed_mmps(50)).wait_for_completed()
        db_connection.delete_request()
        #robot.turn_in_place(degrees(7)).wait_for_completed()
        #robot.turn_in_place(degrees(7)).wait_for_completed()

    ###### 2 - DRUMMMMMMMMMMMMMMMM
    if ROAD == 2:
        robot.turn_in_place(degrees(50)).wait_for_completed()
        robot.drive_straight(distance_mm(40), speed_mmps(50)).wait_for_completed()
        robot.turn_in_place(degrees(4)).wait_for_completed()
        robot.drive_straight(distance_mm(30), speed_mmps(50)).wait_for_completed()
        db_connection.delete_request()

    ###### 3 - DRUMMMMMMM
    if ROAD == 3:
        robot.turn_in_place(degrees(110)).wait_for_completed()
        robot.drive_straight(distance_mm(50), speed_mmps(50)).wait_for_completed()
        robot.turn_in_place(degrees(7)).wait_for_completed()
        db_connection.delete_request()


    # Acest calup se ocupa cu line follower pe curbe
    Process_image_Thread = threading.Thread(target=process_image)
    driveThread = threading.Thread(target=drive, args=(robot,))
    # CameraThread.start()
    Process_image_Thread.start()
    driveThread.start()
    RobotCamera(robot)



cozmo.run_program(line_follower)

