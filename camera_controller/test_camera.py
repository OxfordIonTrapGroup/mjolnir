

import time
import camera_controller

camera = camera_controller.start()

img = camera.get_image()
print(img)
print(type(img))

camera.start_acquisition()

try:
    while(1):
        img = camera.get_image()
        print(img)
        time.sleep(1)
finally:

    camera.stop_acquisition()
    camera.close()






