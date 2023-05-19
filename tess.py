import time

import pytesseract
import PIL
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
# from queue import Queue
#
# # myconfig = r"--psm 6 --oem 3"
# # text = pytesseract.image_to_string(PIL.Image.open("cozmo_pic_2.png"), config=myconfig)
# # print(text)
#
# q = Queue()
#
# q.put(3)
# q.put(5)
# print(q.qsize())
#
# val = q.get()
# print(val)
# print(q.qsize())
#
# while True:
#     pass

def cozmo_program(robot: cozmo.robot.Robot):
    #action1 = robot.drive_straight(distance_mm(50), speed_mmps(25), should_play_anim=False, in_parallel=True)
    # action2 = robot.turn_in_place(degrees(90), in_parallel=True)
    # action2.wait_for_completed()
    robot.drive_wheels(25,50)
    time.sleep(10)
    #cozmo.logger.info("action2 = %s", action2)
    #action1.wait_for_completed()
    #cozmo.logger.info("action1 = %s", action1)

cozmo.run_program(cozmo_program)