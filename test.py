# match_mask_color = (255,) * 5
#
# print(match_mask_color)
# !/usr/bin/env python3

# Copyright (c) 2016 Anki, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Hello World

Make Cozmo say 'Hello World' in this simple Cozmo SDK example program.
'''

import cozmo
import cv2 as cv
import matplotlib.pylab as plt
import math
import numpy as np
from PIL import Image

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


def process_image(image):
    # gray_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2GRAY)
    rgb_image = np.array(image.raw_image)
    pil_image = Image.fromarray(rgb_image)
    pil_image.show()
    # rgb_image = np.array(image.raw_image)
    # rgb_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)
    # plt.imshow(rgb_image,cmap='viridis')
    # plt.show()
    # print(rgb_image)
    #HSV_image = np.array(gray_image, cv.COLOR_BGR2HSV)
    #plt.imshow(HSV_image)
    #plt.show()


    #Blurrr

    # Aplică filtrul Gaussian pentru a estompa imaginea
    blur_image = cv.GaussianBlur(rgb_image, (0, 0), 5)

    # Calculează diferența dintre imaginea originală și imaginea estompată
    high_pass = cv.absdiff(rgb_image, blur_image)

    # Însumează imaginea originală cu imaginea diferență pentru a obține imaginea fără umbre
    shadow_free_image = cv.add(rgb_image, high_pass)

    pil_image = Image.fromarray(shadow_free_image)
    pil_image.show()



    canny_image = cv.bitwise_not(shadow_free_image)
    pil_image = Image.fromarray(canny_image)
    pil_image.show()
    canny_image = cv.Canny(canny_image, 100, 255)

    take_half_image = make_half_image(canny_image)
    lines = cv.HoughLinesP(take_half_image, 1, np.pi / 100, 10, minLineLength=10, maxLineGap=4)

    print(lines)
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv.line(take_half_image, (x1, y1), (x2, y2), (255, 255, 255), 6)

    plt.imshow(take_half_image)
    plt.show()
    #plt.imshow(canny_image)
    # plt.show()
    # plt.imshow(take_half_image)
    #plt.show()
    left = []
    right = []

    height, width = take_half_image.shape
    if len(lines) == 0:
        steering_angle = 0
    else:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            #if x1 == x2:
            if abs(x1-x2)<20:
                continue
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_int = parameters[1]

            if slope < 0:
                if x1 < width / 2 and x2 < width / 2:
                    left.append((slope, y_int))
                    cv.line(take_half_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
            else:
                if x1 > width / 2 and x2 > width / 2:
                    right.append((slope, y_int))
                    cv.line(take_half_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        lanes = []
        plt.imshow(take_half_image)
        plt.show()
        #print(left)
        if len(left) > 0:
            left_avg = np.average(left, axis=0)
            lanes.append(make_points(take_half_image, left_avg))
        #print(left_avg)
        #print(right)
        if len(right) > 0:
            right_avg = np.average(right, axis=0)
            lanes.append(make_points(take_half_image, right_avg))
        #print(right_avg)
        #cv.line(take_half_image, (lanes[0][0],lanes[0][1] ), (lanes[0][2],lanes[0][3] ), (255, 0, 0), 5)
        #cv.line(take_half_image, (lanes[1][0], lanes[1][1]), (lanes[1][2], lanes[1][3]), (255, 0, 0), 5)
        print(lanes)

        if len(lanes) > 1:
            x1, _, left_x2, _ = lanes[0]
            x2, _, right_x2, _ = lanes[1]
            x_offset = (left_x2 + right_x2) / 2 - width / 2
            y_offset = int(height / 2)
            print("2 linii detectate")

            angle_radian = math.atan(x_offset / y_offset)
            print(x_offset/y_offset)
            print(angle_radian)
            print(math.degrees(angle_radian))
            avg_x = int ((left_x2 + right_x2) / 2)
            angle_degree = int(angle_radian * 180.0 / math.pi)
            #cv.line(take_half_image, (avg_x, int(height/2)), (int(width/2), height), (255, 0, 0), 5)
            steering_angle = angle_degree #% 90

            plt.imshow(take_half_image)
            plt.show()
        else:
            if len(lanes) == 1:
                x1, _, x2, _ = lanes[0]
                x_offset = x2 - x1
                y_offset = int(height / 2)
                print("1 linie detectate")
                angle_radian = math.atan(x_offset / y_offset)
                angle_degree = int(angle_radian * 180.0 / math.pi)

                #cv.line(take_half_image, (x2, int(height / 2)), (int(width / 2), height), (255, 0, 0), 5)
                steering_angle = angle_degree #% 90
            else:
                print("0 linii detectate")
                steering_angle = 0

    if 20 > abs(steering_angle) > 0:
         steering_angle = 0

    return steering_angle


def cozmo_program(robot: cozmo.robot.Robot):
    # robot.say_text("Andrei is going home").wait_for_completed()
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False

    #robot.set_head_angle(cozmo.robot.MIN_HEAD_ANGLE,in_parallel=True).wait_for_completed()
    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
    while True:
        latest_image = robot.world.latest_image
        if latest_image:
            break

    if latest_image:
    # rgb_image = np.array(latest_image.raw_image)
    #     print(rgb_image.shape)
    #     if np.all(rgb_image[:, :, 0] == rgb_image[:, :, 1]) and np.all(rgb_image[:, :, 0] == rgb_image[:, :, 2]):
    #         print("Imagine alb-negru")
    #     else:
    #         print("Imagine color")
        # rgb_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)
        # plt.imshow(rgb_image,cmap='viridis')
        # plt.show()
        # gray_image = cv.cvtColor(latest_image, cv.COLOR_RGB2GRAY)
        # print(CameraImages.qsize())
        angle = process_image(latest_image)
        print(angle)
    # plt.imshow(latest_image.raw_image)
    # plt.show()

    robot.camera.image_stream_enabled = False


cozmo.run_program(cozmo_program)
