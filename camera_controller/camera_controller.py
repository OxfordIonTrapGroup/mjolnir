from . import uc480
# print("imported uc480", flush=True)
import numpy as np
import ctypes
import collections
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


class ThorlabsCCD:
    # Servo polling period in seconds
    _POLL_PERIOD = 0.05

    def __init__(self, id_=None, framebuffer_len=100):
        self.camera_id = id_ if id_ is not None else 0
        self._library_init()

        self.quit = False
        self.dead = False
        self.connected = False

        self.acquisition_enabled = False

        self.auto_exposure = True
        self.exposure = None
        self.exposure_min = None
        self.exposure_max = None
        self.exposure_inc = None

        self._frame_buffer = collections.deque([], framebuffer_len)
        self._frame_call_list = []

        # connect to camera
        self._connect()

        # Start image acquisition thread
        t = Thread(target=self._acquisition_thread, daemon=True)
        t.start()

    def _library_init(self):
        self.c = uc480.uc480()

    def _library_cleanup(self):
        del self.c._lib

    def _connect(self):
        """Connect to camera"""
        self.c.connect(self.camera_id)
        self._get_sensor_info()
        self._get_exposure_params()
        self._get_aoi()
        self._get_aoi_absolute()
        self.connected = True

    def _disconnect(self):
        self.c.disconnect()
        self.connected = False

    def _reconnect(self):
        self._disconnect()
        self._library_cleanup()
        self._library_init()
        self._connect()

    def _get_n_cameras(self):
        """Returns the number of cameras connected to the system"""
        n_cams = ctypes.c_int()
        self.c.call("is_GetNumberOfCameras", ctypes.pointer(n_cams))
        return n_cams.value

    def _acquisition_thread(self):
        while self.quit is False:
            if self.acquisition_enabled:
                try:
                    # print("waiting", flush=True)
                    im = timeout(1)(self.c.acquire)(native=True)
                    # print("acquired!", flush=True)
                except (MyTimeoutError, uc480.uc480Error) as e:
                    self._reconnect()
                else:
                    im = self._crop_to_aoi(im)
                    self._frame_buffer.append(im)
                    for f in self._frame_call_list:
                        f(im)

                    # print("enabled {}".format(self.acquisition_enabled), flush=True)
                    # Update the exposure if necessary
                    if self.auto_exposure:
                        # print("try auto", flush=True)
                        timeout(1)(self.handle_auto_exposure)(im)

            # print("sleeping", flush=True)
            time.sleep(self._POLL_PERIOD)
            # print("slept", flush=True)
        self.dead = True

    def start_acquisition(self, single=False):
        """Turn on auto acquire"""
        def acquire_single_cb(_):
            """image is passed to callback and not used"""
            self.stop_acquisition()
            self.deregister_callback(acquire_single_cb)

        if single:
            self.register_callback(acquire_single_cb)
        self.acquisition_enabled = True

    def stop_acquisition(self):
        """Turn off auto acquire"""
        self.acquisition_enabled = False

    def _get_exposure_params(self):
        self.exposure = self.c.get_exposure()
        self.exposure_min, self.exposure_max, \
            self.exposure_inc = self.c.get_exposure_limits()

    def set_exposure_time(self, exposure_time):
        """Set the CCD exposure time in seconds"""
        time_ms = exposure_time*1000
        self.set_exposure_ms(time_ms)

    def set_exposure_ms(self, exposure_time):
        """Set the camera exposure time in ms"""
        self.c.set_exposure(exposure_time)
        self.exposure = exposure_time

    def set_auto_exposure(self, auto_exposure):
        """Enable/disable auto exposure"""
        self.auto_exposure = auto_exposure

    def handle_auto_exposure(self, image):
        """Reads the most recent image and updates exposure"""
        # print("got to exposure handler", flush=True)
        pixel_max = np.amax(image)

        # print("autoexpose", flush=True)
        if pixel_max > auto_exposure_max_threshold:
            exposure = np.maximum(self.exposure*0.9, self.exposure_min)
            # if exposure == self.exposure:
                # print("warning: min exposure reached", flush=True)
            # print("reducing exposure", flush=True)
            self.set_exposure_ms(exposure)

        elif pixel_max < auto_exposure_min_threshold:
            exposure = np.minimum(self.exposure*1.1, self.exposure_max)
            # if exposure == self.exposure:
                # print("warning: max exposure reached", flush=True)
            # print("increasing exposure", flush=True)
            self.set_exposure_ms(exposure)

    def set_image_region(self, hStart, hEnd, vStart, vEnd, **kwargs):
        """Set the CCD region to read out.
        The region is 0 indexed and inclusive, so the valid ranges for hStart
        is 0 ... self.ccd_width - 1 etc."""
        if kwargs:
            print("binning is not supported")
        self.aoi_x = self._clip_to_grid(int(hStart), 0, self.ccd_width, 4)
        self.aoi_y = self._clip_to_grid(int(vStart), 0, self.ccd_height, 2)
        self.aoi_width = self._clip_to_grid(int(1+hEnd-hStart), 32, self.ccd_width, 4)
        self.aoi_height = self._clip_to_grid(int(1+vEnd-vStart), 4, self.ccd_height, 2)

        self._set_aoi(self.aoi_x, self.aoi_y, self.aoi_height, self.aoi_width)
        # exposure settings will change
        self._get_exposure_params()

    def _clip_to_grid(self, in_val, min_, max_, grid):
        out_val = np.clip(in_val, min_, max_)
        # integer maths does the rounding
        out_val = out_val // grid * grid
        return out_val

    def _set_aoi(self, posx, posy, width, height, absolute=True):
        """Set Area Of Interest according to Thorlabs spec
        If absolute is True, the memory location of the aoi within the image
        matches its actual position
        """
        rectAOI = uc480.IS_RECT()
        rectAOI.s32X = posx
        rectAOI.s32Y = posy
        rectAOI.s32Width = width
        rectAOI.s32Height = height

        self.aoi_absolute = absolute
        if self.aoi_absolute:
            rectAOI.s32X |= uc480.IS_AOI_IMAGE_POS_ABSOLUTE
            rectAOI.s32Y |= uc480.IS_AOI_IMAGE_POS_ABSOLUTE

        self.c.call("is_AOI", self.c._camID, uc480.IS_AOI_IMAGE_SET_AOI,
            ctypes.pointer(rectAOI), ctypes.sizeof(rectAOI))

    def _get_aoi(self):
        """Gets the aoi parameters, sets internal values and returns them"""
        rectAOI = uc480.IS_RECT()
        self.c.call("is_AOI", self.c._camID, uc480.IS_AOI_IMAGE_GET_AOI,
            ctypes.pointer(rectAOI), ctypes.sizeof(rectAOI))

        self.aoi_width = rectAOI.s32Width
        self.aoi_height = rectAOI.s32Height

        # the following is agnostic of whether the aoi mode is absolute or not
        self.aoi_x = rectAOI.s32X & ~uc480.IS_AOI_IMAGE_POS_ABSOLUTE
        self.aoi_y = rectAOI.s32Y & ~uc480.IS_AOI_IMAGE_POS_ABSOLUTE

        return self.aoi_x, self.aoi_y, self.aoi_width, self.aoi_height

    def _get_aoi_absolute(self):
        x_abs = ctypes.c_int32()
        self.c.call("is_AOI", self.c._camID, uc480.IS_AOI_IMAGE_GET_POS_X_ABS,
            ctypes.pointer(x_abs), ctypes.sizeof(x_abs))

        y_abs = ctypes.c_int32()
        self.c.call("is_AOI", self.c._camID, uc480.IS_AOI_IMAGE_GET_POS_Y_ABS,
            ctypes.pointer(y_abs), ctypes.sizeof(y_abs))

        if np.all([x_abs, y_abs]):
            self.aoi_absolute = True
        elif not np.any([x_abs, y_abs]):
            self.aoi_absolute = False
        else:
            print("either both or neither axis of aoi should be absolute")
        return self.aoi_absolute

    def _crop_to_aoi(self, image):
        """crop the image to the size of the aoi"""
        if self.aoi_width == self.ccd_width \
                and self.aoi_height == self.ccd_height:
            # shortcut for when the aoi is the whole sensor
            return image

        x_start = self.aoi_x if self.aoi_absolute else 0
        y_start = self.aoi_y if self.aoi_absolute else 0
        x_end = self.aoi_width + x_start
        y_end = self.aoi_height + y_start
        return image[y_start:y_end, x_start:x_end]

    def _get_sensor_info(self):
        # get sensor info
        pInfo = uc480.SENSORINFO()
        self.c.call("is_GetSensorInfo", self.c._camID, ctypes.pointer(pInfo))
        self.ccd_width = pInfo.nMaxWidth
        self.ccd_height = pInfo.nMaxHeight

    def register_callback(self, f):
        """Register a function to be called from the acquisition thread for each
        new image"""
        self._frame_call_list.append(f)

    def deregister_callback(self, f):
        if f in self._frame_call_list:
            self._frame_call_list.remove(f)

    def get_image(self):
        """Returns the oldest image in the buffer as a numpy array, or None if
        no new images"""
        if len(self._frame_buffer) == 0:
            return None
        return self._frame_buffer.popleft()

    def get_all_images(self):
        """Returns all of the images in the buffer as an array of numpy arrays,
        or None if no new images"""
        if len(self._frame_buffer):
            ims = []
            while len(self._frame_buffer) > 0:
                ims.append(self._frame_buffer.popleft())
        else:
            ims = None
        return ims

    def ping(self):
        return True

    def close(self):
        self.quit = True
        while not self.dead:
            time.sleep(0.01)
        self._disconnect()
        self._library_cleanup()

    def __del__(self):
        self.close()
