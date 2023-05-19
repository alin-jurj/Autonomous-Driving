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


def verifying_rgb_image(rgb_image):
    rgb_image = np.array(rgb_image)
    if np.all(rgb_image[:, :, 0] == rgb_image[:, :, 1]) and np.all(rgb_image[:, :, 0] == rgb_image[:, :, 2]):
        return 0
    else:
        return 1
def cozmo_program(robot: cozmo.robot.Robot):
    # Abilitați camera robotului
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True

    while True:
        # Așteptați până când se primește o imagine
        latest_image = robot.world.latest_image
        while latest_image is not None:

            # convertim imaginea in format numpy

            image_np = latest_image.raw_image
            if(verifying_rgb_image(image_np)==1):
            # Afișăm imaginea
                plt.imshow(image_np)
                plt.show()
                break
            latest_image = robot.world.latest_image


cozmo.run_program(cozmo_program)