import os
import sys
import numpy as np
import logging
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

logger = logging.getLogger(__name__)

class tsi:
    """Main class for communication with one of Thorlabs' tsi cameras.
    """
    def __init__(self):
        """Constructor.

        Takes no arguments but tries to automatically configure the system path and
        creates a list of all connected cameras.
        """
        self._cam_list = []
        self._sn = None
        self._model = None
        self._swidth = 0
        self._sheight = 0
        self._rgb = 0

        self._image = None
        self._imgID = None

        self.configure_path()
        self.get_cameras()

    def configure_path(self):
        """Configure the system path to include the directory containing the dlls.
        """
        logger.debug("Configure system path..")
        
        is_64bits = sys.maxsize > 2**32
        
        # change the value below to define dll system path
        path_to_dlls = r"C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support"\
                       "\Scientific_Camera_Interfaces-Rev_G\Scientific Camera Interfaces\SDK"\
                       "\Python Compact Scientific Camera Toolkit\dlls"

        if is_64bits:
            path_to_dlls += r"\64_lib"
        else:
            path_to_dlls += r"\32_lib"

        os.environ['PATH'] = path_to_dlls + os.pathsep + os.environ['PATH']

        logger.info((path_to_dlls + " added to system path"))

    def get_cameras(self):
        """Finds the connected tsi cameras.
        """
        with TLCameraSDK() as sdk:
            available_cameras = sdk.discover_available_cameras()
        nCams = len(available_cameras)
        logger.info(("Found %d tsi camera(s)" % nCams))
        if nCams > 0:
            self._cam_list = available_cameras

    def connect(self, sn):
        """Connect to the camera with the given serial number.
        """
        self.sdk = TLCameraSDK()
        self.camera = self.sdk.open_camera(sn)

        self._sn = self.camera.serial_number
        self._model = self.camera.model

        self._swidth = self.camera.sensor_width_pixels
        self._sheight = self.camera.sensor_height_pixels
        self._rgb = (self.camera.camera_sensor_type == 1)
        self._bitsperpixel = self.camera.bit_depth
        logger.info(("Sensor: %d x %d pixels, RGB = %d, %d bits/px" %
                     (self._swidth, self._sheight, self._rgb, self._bitsperpixel)))

        self.camera.operation_mode = 0 # software-triggered
        self.camera.frames_per_trigger_zero_for_unlimited = 1  # single frame per trigger
        self.camera.is_frame_rate_control_enabled = False # disable frame rate control

        self.fpsmin = self.camera.frame_rate_control_value_range.min # min fps
        self.fpsmax = self.camera.frame_rate_control_value_range.max # max fps

        self.expmin = self.camera.exposure_time_range_us.min / 1000. # exposure min in ms
        self.expmax = self.camera.exposure_time_range_us.max / 1000. # exposure max in ms
        logger.info(("Valid exposure times: %fms to %fms" % (self.expmin, self.expmax)))
        self.set_exposure(self.expmin) # set exposure to min value

        self.camera.image_poll_timeout_ms = 10000  # 10 second polling timeout

        self.camera.arm(2) # arm the camera with a two-frame buffer
        
    def disconnect(self):
        """Disconnect a currently connected camera.
        """
        self.camera.disarm()
        self.camera.dispose()
        self.sdk.dispose()
    
    def stop(self):
        """Same as disconnect.
        """
        self.disconnect()

    def get_pixel_width(self):
        """Gets pixel width in microns.
        """
        return self.camera.sensor_pixel_width_um

    def get_sensor_size(self):
        """Returns the sensor size in pixels as a tuple: (width, height)

        If not connected yet, it returns a zero tuple.
        """
        return self._swidth, self._sheight

    def get_image_size(self):
        """Returns the image size in pixels as a tuple: (width, height)
        """
        return self.camera.image_width_pixels, self.camera.image_height_pixels
    
    def get_frame_rate_limits(self):
        """Returns the frame rate limits in fps (min, max, increment=0.1).
        """
        return self.camera.frame_rate_control_value_range.min, self.camera.frame_rate_control_value_range.max, 0.1
    
    def set_gain(self, gain):
        """Set the hardware gain.

        :param int gain: New gain setting.
        """
        self.camera.gain = int(gain)

    def get_gain(self):
        """Returns current gain setting.
        """
        return self.camera.gain

    def get_gain_limits(self):
        """Returns gain limits (min, max, increment=1).
        """
        return self.camera.gain_range.min, self.camera.gain_range.max, 1
    
    def set_blacklevel(self, blck):
        """Set blacklevel.

        :param int blck: New blacklevel setting.
        """
        self.camera.black_level = int(blck)

    def get_blacklevel_limits(self):
        """Returns blacklevel limts (min, max).
        """
        return self.camera.black_level_range.min, self.camera.black_level_range.max

    def set_exposure(self, exp):
        """Set exposure time in milliseconds.

        :param int exp: New exposure time.
        """
        self.camera.exposure_time_us = int(exp * 1000)

    def get_exposure(self):
        """Returns current exposure time in milliseconds.
        """
        return self.camera.exposure_time_us / 1000.

    def get_exposure_limits(self):
        """Returns the supported limits for the exposure time in ms (min, max, increment).
        """
        _min = self.camera.exposure_time_range_us.min / 1000. # exposure min in ms
        _max = self.camera.exposure_time_range_us.max / 1000. # exposure max in ms
        
        if self.camera.usb_port_type == 2:
            inc = 1000 / (1125 * self.camera.frame_rate_control_value_range.max) # exposure step size for usb3.0
        else:
            inc = 1000 / (582 * self.camera.frame_rate_control_value_range.max) # exposure step size for usb2.0
        
        return _min, _max, inc

    def set_roi(self, posx, posy, width, height):
        """Set the ROI.
        """
        self.camera.roi = (posx, posy, width + posx, height + posy)

    def get_roi(self):
        """Returns the current ROI.
        """
        return self.camera.roi

    def arm(self, N=2):
        """Arms the camera with an N-frame buffer.
        """
        self.camera.arm(N)

    def trigger(self):
        """Issues a software trigger.
        """
        self.camera.issue_software_trigger()

    def disarm(self):
        """Disarms the camera.
        """
        self.camera.disarm()

    def get_is_armed(self):
        """Returns True if the camera is armed, False if unarmed.
        """
        return self.camera.is_armed

    def acquire(self, N=1, native=False):
        """Synchronously captures some frames from the camera using the current settings and returns the averaged image.

        :param int N: Number of frames to acquire (>= 1).
        :returns: Averaged image.
        """
        logger.debug(("acquire %d frames" % N))
        data = None
        
        for i in range(int(N)):
            self.camera.issue_software_trigger()
            logger.debug("  wait for data..")
            frame = self.camera.get_pending_frame_or_null()
            self._image = frame.image_buffer
            self._imgID = frame.frame_count
            logger.debug("  read data..")
            if data is None:
                if native:
                    data = self._image
                else:
                    data = self._image.astype(float)
            else:
                data = data + self._image
        if not native:
            data = data / float(N)

        return data
    
    def acquireBinned(self, N=1):
        """Record N frames from the camera using the current settings and return fully binned 1d arrays averaged over the N frames.

        :param int N: Number of images to acquire.
        :returns: - Averaged 1d array fully binned over the x-axis.
                  - Averaged 1d array fully binned over the y-axis.
                  - Maximum pixel intensity before binning, e.g. to detect over illumination.
        """
        data = self.acquire(N)
        return np.sum(data, axis=0), np.sum(data, axis=1), np.amax(data)

    def acquireMax(self, N=1):
        """Record N frames from the camera using the current settings and return the column / row with the maximum intensity.

        :param int N: Number of images to acquire.
        :returns: - Column with maximum intensity (1d array).
                  - Row with maximum intensity (1d array).
        """
        data = self.acquire(N)
        return data[np.argmax(np.data, axis=0), :], data[:, np.argmax(np.data, axis=1)]
