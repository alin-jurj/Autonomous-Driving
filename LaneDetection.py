import math
import time

import cozmo
import numpy as np
import matplotlib.pylab as plt
import cv2 as cv

def points_on_lines(img):
    result = np.where(img == 255)
    listOfCoordinates = list(zip(result[0], result[1]))

    return listOfCoordinates

#image: cozmo.world.CameraImage
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255 #* channel_count

    cv.fillPoly(mask, vertices, match_mask_color)

    masked_image = cv.bitwise_and(img, mask)

    return masked_image
def calculate_distance(x,y):
    return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
def calculate_minimum_distance(Coord_Curve1,Coord_Curve2):
    min = 999
    #print(Coord_Curve1)
    #print(Coord_Curve2)
    for i2 in range(0, int(len(Coord_Curve2)/2)):
        for i1 in range(0, len(Coord_Curve1)):
            # print(Coord_Curve1[i1])
            # print(first)
            distance = calculate_distance(Coord_Curve1[i1], Coord_Curve2[i2])
            if distance < min:
                min = distance
                point1 = i1
                point2 = i2
    return [min, point1, point2]


#robot:cozmo.robot.Robot = None
def Canny_treshold():
    # robot.camera.image_stream_enabled = True
    # robot.say_text("We are going").wait_for_completed()
    # print("taking a picture...")
    #pic_filename = "cozmo_pic_" + str(int(time.time())) + ".png"
    # latest_image = robot.world.latest_image
    image = cv.imread('curba_poza.png')
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    height = image.shape[0]
    width = image.shape[1]
    lane_vertices=[
        (width/2-40,160), #(0,height) 175
        (width/2, 0),
        (width/2+50, 160) #(width,height)
    ]
    left_margin_vertices=[
        (15,153),
        (40,75),
        (127, 106),
        (120,161)

    ]
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    canny_image = cv.Canny(gray_image, 180, 255) # 150 200
    lane_cropped_image = region_of_interest(canny_image,
                np.array([lane_vertices], np.int32),)
    left_margin_cropped_image = region_of_interest(canny_image,
                                       np.array([left_margin_vertices], np.int32), )

    plt.imshow(canny_image)
    plt.show()
    lane_coordinates = points_on_lines(lane_cropped_image)
    lane_coordinates.reverse()
    # for i1 in range(0, len(lane_coordinates)):
    #     if i1[0]-
    left_margin_coordinates = points_on_lines(left_margin_cropped_image)
    left_margin_coordinates.reverse()

    leg1= calculate_minimum_distance(left_margin_coordinates, lane_coordinates)
    print(leg1)
    y1 = left_margin_coordinates[leg1[1]][0]
    print(left_margin_coordinates[leg1[1]])
    print(lane_coordinates[leg1[2]])
    y2 = lane_coordinates[leg1[2]][0]

    diviziune = max(y1, y2)
    diviziune = math.floor(diviziune/10)
    theta = 90 - math.floor(abs(y2-y1)/diviziune) * 10

    print(theta)
    # print(len(lane_coordinates))

    # leg2 = calculate_distance(left_margin_coordinates[leg1[1]], lane_coordinates[int(len(lane_coordinates)/2)])
    # d = calculate_distance(lane_coordinates[0], lane_coordinates[int(len(lane_coordinates)/2)])

    # #print(lef_margin_coordinates[0][1])


    plt.imshow(lane_cropped_image)
    plt.show()
    plt.imshow(left_margin_cropped_image)
    plt.show()




    # if(latest_image):
        #image = cv.cvtColor(np.array(latest_image.raw_image), cv.COLOR_BGR2RGB)
        # latest_image.raw_image.convert('L').save("roaddd4.jpg")
        #print(image.shape)



Canny_treshold()

#cozmo.run_program(Canny_treshold)
