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

def change_perspective(img):
    cbl = (53,212)
    cbr = (316, 212)
    ctl = (92, 75)
    ctr = (254,88)
    point1 = np.float32([ctl,cbl,ctr,cbr])
    point2  = np.float32([[0,0],[0,255],[320,0],[320,255]])
    changed_matrix = cv.getPerspectiveTransform(point1, point2)
    transformed_image = cv.warpPerspective(img, changed_matrix,(320,255))
    plt.imshow(transformed_image)
    plt.show()
    return transformed_image
def histogram(img):

    histogram = np.sum(img, axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    print(("Histogram= " + str(histogram)))

    print(left_base)
    print(midpoint)
    print(right_base)
    ##Sliding window

    y=200
    lx=[]
    rx=[]

    mask = img.copy()

    while y>0:
        ##LEFT TRESHOLD
        msk = mask[y-40:y,left_base-50:left_base+50]
        contours, _ = cv.findContours(msk,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            M = cv.moments(contour)

            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"] / M["m00"])
                lx.append(left_base-50+cx)
                left_base = left_base-50+cx

        ##RIGHT TRESHOLD
        msk = mask[y - 40:y, right_base - 50:right_base + 50]
        contours, _ = cv.findContours(msk, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            M = cv.moments(contour)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lx.append(right_base - 50 + cy)
                left_base = right_base - 50 + cy
        cv.rectangle(img, (left_base-50,y), (left_base+50,y-40),(255,255,255),2)
        cv.rectangle(img, (right_base - 50, y), (right_base + 50, y - 40), (255, 255, 255), 2)
        y = y-40
        plt.imshow(img)
        plt.show()

        return img
#robot:cozmo.robot.Robot = None
def nothing(x):
    pass
def Canny_treshold():

    # robot.camera.image_stream_enabled = True
    # robot.say_text("We are going").wait_for_completed()
    # print("taking a picture...")
    #pic_filename = "cozmo_pic_" + str(int(time.time())) + ".png"
    # latest_image = robot.world.latest_image
    image = cv.imread('curba_ok.png')
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
    #for i1 in range(0, len(lane_coordinates)):
    #    if i1[0]-
    left_margin_coordinates = points_on_lines(left_margin_cropped_image)
    left_margin_coordinates.reverse()

   # leg1= calculate_minimum_distance(left_margin_coordinates, lane_coordinates)
    #print(leg1)
    #y1 = left_margin_coordinates[leg1[1]][0]
    #print(left_margin_coordinates[leg1[1]])
    #print(lane_coordinates[leg1[2]])
    #y2 = lane_coordinates[leg1[2]][0]

    #diviziune = max(y1, y2)
    #diviziune = math.floor(diviziune/10)
    #theta = 90 - math.floor(abs(y2-y1)/diviziune) * 10

   # print(theta)
    print(len(lane_coordinates))

    #leg2 = calculate_distance(left_margin_coordinates[leg1[1]], lane_coordinates[int(len(lane_coordinates)/2)])
    #d = calculate_distance(lane_coordinates[0], lane_coordinates[int(len(lane_coordinates)/2)])

    #print(lef_margin_coordinates[0][1])

    img = change_perspective(canny_image)
    histogram(img)
    # plt.imshow(lane_cropped_image)
    # plt.show()
    # plt.imshow(left_margin_cropped_image)
    # plt.show()




    # if(latest_image):
        #image = cv.cvtColor(np.array(latest_image.raw_image), cv.COLOR_BGR2RGB)
        # latest_image.raw_image.convert('L').save("roaddd4.jpg")
        #print(image.shape)



Canny_treshold()

#cozmo.run_program(Canny_treshold)
