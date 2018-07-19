"""An example of how to take a continuous stream of images using the AndorEmccd class"""
import time
import matplotlib.pyplot as plt
from camera_controller.camera_controller import ThorlabsCCD
print("imported", flush=True)

cam = ThorlabsCCD()
print("connected!", flush=True)

print(cam.ccd_width, cam.ccd_height)

cam.set_exposure_time(0.01)
print("set exposure", flush=True)

cam.start_acquisition()
print("starting acquisition", flush=True)

n_acquired = 0
t_start = time.time()
while n_acquired < 5:
    imVec = cam.get_all_images()
    if imVec is None:
        continue
    n_acquired += len(imVec)
print("n_acquired = {}. Frame rate = {} /s".format(n_acquired, n_acquired/(time.time()-t_start)))

im=None
while im is None:
    im = cam.get_image()
    time.sleep(0.05)
print(im.shape)

cam.set_image_region(0,512, 0, 256)
print("set region")

im=None
while im is None:
    im = cam.get_image()
    time.sleep(0.05)
print(im.shape)
