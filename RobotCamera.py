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

CameraImages = Queue()
Steering_Angles = Queue()


classes = {
    0: 'Yield',
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
    image = image[:int(height-70),width-100:,:]
    return image

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

        # blur_image = cv.GaussianBlur(rgb_image, (0, 0), 7)
        # high_pass = cv.absdiff(rgb_image, blur_image)
        # shadow_free_image = cv.add(rgb_image, high_pass)

        hsv = cv.cvtColor(rgb_image, cv.COLOR_RGB2HSV)

        # lower_green = np.array([26, 121, 66])
        # upper_green = np.array([102, 153, 247])

        # lower_green = np.array([56, 40, 69])
        # upper_green = np.array([85, 128, 128])

        # lower_green = np.array([16, 45, 64])
        # upper_green = np.array([89, 128, 223])
        # cele de sus sunt bune pentru lumina caldaa care nu e pe impuls.
        # lower_green = np.array([26, 57, 66])
        # upper_green = np.array([96, 149, 190])
        # lower_green = np.array([50, 0, 64])
        # upper_green = np.array([96, 246, 243])

        lower_green = np.array([0,23,116])
        upper_green = np.array([110,116,255])

        # lower_green = np.array([60, 64, 128])
        # upper_green = np.array([104, 94, 212])
        green_mask = cv.inRange(hsv, lower_green, upper_green)

        #green_mask = cv.erode(green_mask, (3, 3), iterations=2)

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
                steering_angle = angle_degree #% 90
            else:
                if len(lanes) == 1:
                    x1, _, x2, _ = lanes[0]
                    x_offset = x2 - x1
                    y_offset = int(height/2)

                    angle_radian = math.atan(x_offset / y_offset)
                    angle_degree = int(angle_radian * 180.0 / math.pi)
                    steering_angle = angle_degree #% 90
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
                robot.drive_wheels(50,15)
            else:
                if steering > 40:
                    robot.drive_wheels(38,12)
                else:
                    robot.drive_wheels(50,25)
            #robot.drive_wheels(50,25)
        if steering < 0:
            if steering < -60:
                robot.drive_wheels(15,50)
            else:
                if steering < -40:
                    robot.drive_wheels(12,38)
                else:
                    robot.drive_wheels(25,50)
        if steering == 0:
            robot.drive_wheels(18,18)
        # if action1 is not None:
        #     action1.abort()
        #
        # action1 = robot.drive_straight(distance_mm(50), speed_mmps(25), should_play_anim=False, in_parallel=True)
        # action2 = robot.turn_in_place(degrees(steering), in_parallel=True)
        # action2.wait_for_completed()
        #robot.turn_in_place(degrees(steering)).wait_for_completed()

     return


def RobotCamera(robot: cozmo.robot.Robot = None):
    # robot.say_text("Andrei is going home").wait_for_completed()
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
        # gray_image = cv.cvtColor(latest_image, cv.COLOR_RGB2GRAY)
        # print(CameraImages.qsize())
        while True:
            latest_image = robot.world.latest_image
            if latest_image:
                latest_image_array = np.array(latest_image.raw_image)
                if verifying_rgb_image(latest_image_array):
                    break

    robot.camera.image_stream_enabled = False


def line_follower(robot: cozmo.robot.Robot):
    # CameraThread = threading.Thread(target=RobotCamera, args=(robot,))
    # robot.camera.image_stream_enabled = True
    # robot.camera.color_image_enabled = True
    # robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
    # model = load_model("model5.h5")
    # while True:
    #     latest_image = robot.world.latest_image
    #     if latest_image:
    #         latest_image_array = np.array(latest_image.raw_image)
    #         if verifying_rgb_image(latest_image_array):
    #             break
    # #rgb_image = cv.cvtColor(np.array(latest_image.raw_image), cv.COLOR_BGR2RGB)
    #
    # imag = region_of_interest(np.array(latest_image.raw_image))
    # imag = cv.resize(imag,(70,70))
    #
    # plt.imshow(imag)
    # plt.show()
    # data = []
    # data.append(np.array(imag))
    # images = np.array(data)
    #
    # pred = model.predict(images)
    # classes_x = np.argmax(pred, axis=1)
    # print(pred)
    # print(classes[classes_x[0]])
    # print(imag.shape)
    # Acest calup se ocupa cu line follower pe curbe
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


# try:
cozmo.run_program(line_follower)
    # cozmo.run_program(run, use_viewer=True, force_viewer_on_top=True)
# except SystemExit as e:
#     print('exception = "%s"' % e)
#     # ONLY FOR TESTING PURPOSES
#     print('\nGoing on without Cozmo: for testing purposes only!', 'red')
#     line_follower(None)
