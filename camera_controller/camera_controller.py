


from . import uc480
import numpy as np
import ctypes
import time

from threading import Thread
import functools

# Thresholds for increasing and decreasing exposure
auto_exposure_max_threshold = 220
auto_exposure_min_threshold = 150


class MyTimeoutError(Exception):
    pass

def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [MyTimeoutError('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco


class CameraController:
    # Servo polling period in seconds
    _POLL_PERIOD = 0.01

    def __init__(self):

        self.c = uc480.uc480()

        self.quit = False
        self.connected = False

        # Does this controller automatically acquire images?
        self.acquisition_enabled = False

        self._image = None
        self.new_image = False

        self.auto_exposure = True
        self.exposure = 0
        self.exposure_min = 0
        self.exposure_max = 1
        self.exposure_inc = 0.1

    def _acquisition_thread_exec(self):

        while self.quit is False:

            if self.acquisition_enabled:
                self.acquire()

            time.sleep(self._POLL_PERIOD)

    def get_n_cameras(self):
        """Returns the number of cameras connected to the system"""
        n_cams = ctypes.c_int()
        self.c.call("is_GetNumberOfCameras", ctypes.pointer(n_cams))
        return n_cams.value

    def acquire(self):

        n_cams = self.get_n_cameras()

        if n_cams == 0:
            return

        if self.c._camID is None:
            self.connect()

        try:
            self._image = timeout(1)(self.c.acquire)(native=True)
            self.new_image = True

            # Update the exposure if necessary
            if self.auto_exposure:
                 timeout(1)(self.handle_auto_exposure)()

        except (MyTimeoutError, uc480.uc480Error) as e:
            print("Disconnected from camera")
            self.disconnect()
            print("Cleaning up library")
            del self.c._lib
            print("Restarting library")
            self.c = uc480.uc480()

    def connect(self):
        self.c.connect()
        self.exposure = self.c.get_exposure()
        self.exposure_min, self.exposure_max, \
            self.exposure_inc = self.c.get_exposure_limits()
        self.connected = True

    def disconnect(self):
        self.c.disconnect()
        self.connected = False

    def ping(self):
        return True

    def start_acquisition(self):
        """Turn on auto acquire"""
        self.acquisition_enabled = True

    def stop_acquisition(self):
        """Turn off auto acquire"""
        self.acquisition_enabled = False


    def close(self):
        self.quit = True

    def get_image(self):
        """Returns last image"""
        return self._image

    def set_exposure(self, exposure_time):
        """Set the camera exposure time is ms"""
        self.c.set_exposure(exposure_time)
        self.exposure = exposure_time

    def set_auto_exposure(self, auto_exposure):
        """Enable/disable auto exposure"""
        self.auto_exposure = auto_exposure

    def handle_auto_exposure(self):
        """Reads the most recent image and updates exposure"""
        pixel_max = np.max(self._image)

        if pixel_max > auto_exposure_max_threshold:
            exposure = self.exposure*0.9
            if exposure < self.exposure_min:
                exposure = self.exposure_min
            self.set_exposure(exposure)

        elif pixel_max < auto_exposure_min_threshold:
            exposure = self.exposure*1.1
            if exposure > self.exposure_max:
                exposure = self.exposure_max
            self.set_exposure(exposure)





def start():
    import threading
    dev = CameraController()

    t = threading.Thread(target=dev._acquisition_thread_exec)
    t.daemon = True
    t.start()
    return dev

