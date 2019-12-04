"""
.. module: uc480
   :platform: Windows, Linux
.. moduleauthor:: Daniel Dietze <daniel.dietze@berkeley.edu>

..
   This file is part of the uc480 python module.

   The uc480 python module is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The uc480 python module is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with the uc480 python module. If not, see <http://www.gnu.org/licenses/>.

   Copyright 2015 Daniel Dietze <daniel.dietze@berkeley.edu>.
"""
import time
import ctypes
import platform
_linux = (platform.system() == "Linux")
import numpy as np
import logging

from .uc480_h import *


logger = logging.getLogger(__name__)


# ##########################################################################################################
# helper functions
def ptr(x):
    return ctypes.pointer(x)


# ##########################################################################################################
# Error handling
class uc480Error(Exception):
    """uc480 exception class handling errors related to communication with the uc480 camera.
    """
    def __init__(self, error, mess, fname=""):
        """Constructor.

        :param int error: Error code.
        :param str mess: Text message associated with the error.
        :param str fname: Name of function that caused the error (optional).
        """
        self.error = error
        self.mess = mess
        self.fname = fname

    def __str__(self):
        if self.fname != "":
            return self.mess + ' in function ' + self.fname
        else:
            return self.mess


def assrt(retVal, fname=""):
    if not (retVal == IS_SUCCESS):
        raise uc480Error(retVal, "Error: uc480 function call failed! Error code = " + str(retVal), fname)
    return retVal


# ##########################################################################################################
# Camera Class
class uc480:
    """Main class for communication with one of Thorlabs' uc480 cameras.
    """
    def __init__(self):
        """Constructor.

        Takes no arguments but tries to automatically connect to the uc480 library and creates a list of all connected cameras.
        """
        # variables
        self._lib = None
        self._cam_list = create_camera_list(0)
        self._camID = None
        self._swidth = 0
        self._sheight = 0
        self._rgb = 0

        self._image = None
        self._imgID = None

        # library initialization
        # connect to uc480 DLL
        self.connect_to_library()

        # get list of cameras
        self.get_cameras()


    # wrapper around function calls to allow the user to call any library function
    def call(self, function, *args):
        """Wrapper around library function calls to allow the user to call any library function.

        :param str function: Name of the library function to be executed.
        :param mixed args: Arguments to pass to the function.
        :raises uc480Error: if function could not be properly executed.
        """
        logger.debug(("calling %s.." % function))
        func = getattr(self._lib, function, None)
        if func is not None:
            if _linux and function in ["is_RenderBitmap", "is_GetDC", "is_ReleaseDC", "is_UpdateDisplay",
                                       "is_SetDisplayPos", "is_SetHwnd", "is_SetUpdateMode",
                                       "is_GetColorDepth", "is_SetOptimalCameraTiming", "is_DirectRenderer"]:
                logger.warning(("WARNING: Function %s is not supported by this library version.." % function))
            else:
                assrt(func(*args), function)
        else:
            logger.warning(("WARNING: Function %s does not exist in this library version.." % function))


    # use this version if the called function actually returns a value
    def query(self, function, *args):
        """Wrapper around library function calls to allow the user to call any library function AND query the response.

        :param str function: Name of the library function to be executed.
        :param mixed args: Arguments to pass to the function.
        :returns: Result of function call.
        :raises uc480Error: if function could not be properly executed.
        """
        logger.debug(("querying %s.." % function))
        func = getattr(self._lib, function, None)
        if func is not None:
            if _linux and function in ["is_RenderBitmap", "is_GetDC", "is_ReleaseDC", "is_UpdateDisplay",
                                       "is_SetDisplayPos", "is_SetHwnd", "is_SetUpdateMode",
                                       "is_GetColorDepth", "is_SetOptimalCameraTiming", "is_DirectRenderer"]:
                logger.warning(("WARNING: Function %s is not supported by this library version.." % function))
            else:
                return func(*args)
        else:
            logger.warning(("WARNING: Function %s does not exist in this library version.." % function))
            return


    # connect to uc480 DLL library
    def connect_to_library(self, library=None):
        """Establish connection to uc480 library depending on operating system and version. If no library name is given (default), the function looks for

            - **uc480.dll** on Win32
            - **uc480_64.dll** on Win64
            - **libueye_api.so** on Linux (check this, but appeared to be for v4.91)

        :param str library: If not None, try to connect to the given library name.
        """
        logger.debug("Load uc480 library..")

        if library is None:
            if _linux:
                self._lib = ctypes.cdll.LoadLibrary("libueye_api.so")
            else:
                if (platform.architecture()[0] == "32bit"):
                    self._lib = ctypes.cdll.LoadLibrary("uc480.dll")
                else:
                    self._lib = ctypes.cdll.LoadLibrary("uc480_64.dll")
        else:
            self._lib = ctypes.cdll.LoadLibrary(library)

        # get version
        version = self.query("is_GetDLLVersion")
        build = version & 0xFFFF
        version = version >> 16
        minor = version & 0xFF
        version = version >> 8
        major = version & 0xFF
        logger.info(("API version %d.%d.%d" % (major, minor, build)))


    # query number of connected cameras and retrieve a list with CameraIDs
    def get_cameras(self):
        """Queries the number of connected cameras and prints a list with the available CameraIDs.
        """
        nCams = ctypes.c_int()
        self.call("is_GetNumberOfCameras", ptr(nCams))
        nCams = nCams.value
        logger.info(("Found %d camera(s)" % nCams))
        if nCams > 0:
            self._cam_list = create_camera_list(nCams)
            self.call("is_GetCameraList", ptr(self._cam_list))

            for i in range(self._cam_list.dwCount):
                camera = self._cam_list.uci[i]
                logger.info(("Camera #%d: SerNo = %s, CameraID = %d, DeviceID = %d" % (i, camera.SerNo, camera.dwCameraID, camera.dwDeviceID)))


    # connect to camera with given cameraID; if cameraID = 0, connect to first available camera
    def connect(self, ID=0, useDevID=False):
        """Connect to the camera with the given cameraID. If cameraID is 0, connect to the first available camera. When connected, sensor information is read out, image memory is reserved and some default parameters are submitted.

        .. versionchanged:: 11-28-2016

            - Added `useDevID` to enable camera selection via cameraID or deviceID.

        :param int ID: ID of the camera to connect to. Set this to 0 to connect to the first available camera (default).
        :param bool useDevID: Set to True if camera should be identified by deviceID instead. By default (False), cameraID is used.
        """
        # connect to camera
        self._camID = HCAM(ID) if not useDevID else HCAM(ID | IS_USE_DEVICE_ID)
        self.call("is_InitCamera", ptr(self._camID), None)

        # get sensor info
        pInfo = SENSORINFO()
        self.call("is_GetSensorInfo", self._camID, ptr(pInfo))
        self._swidth = pInfo.nMaxWidth
        self._sheight = pInfo.nMaxHeight
        self._rgb = not (pInfo.nColorMode == IS_COLORMODE_MONOCHROME)
        if self._rgb:
            self.call("is_SetColorMode", self._camID, IS_CM_RGB8_PACKED)
            self._bitsperpixel = 24
        else:
            self.call("is_SetColorMode", self._camID, IS_CM_MONO8)
            self._bitsperpixel = 8
        logger.info(("Sensor: %d x %d pixels, RGB = %d, %d bits/px" % (self._swidth, self._sheight, self._rgb, self._bitsperpixel)))

        dblRange = (ctypes.c_double * 3)()
        self.call("is_Exposure", self._camID, IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE, ptr(dblRange), ctypes.sizeof(dblRange))
        logger.info(("Valid exposure times: %fms to %fms in steps of %fms" % (dblRange[0], dblRange[1], dblRange[2])))
        self.expmin, self.expmax, self.expinc = dblRange

        # set default parameters
        self.call("is_ResetToDefault", self._camID)
        self.call("is_SetExternalTrigger", self._camID, IS_SET_TRIGGER_OFF)
        self.call("is_SetGainBoost", self._camID, IS_SET_GAINBOOST_OFF)
        self.call("is_SetHardwareGain", self._camID, 0, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER)
        self.call("is_Blacklevel", self._camID, IS_BLACKLEVEL_CMD_SET_MODE, ptr(ctypes.c_int(IS_AUTO_BLACKLEVEL_OFF)), ctypes.sizeof(ctypes.c_int))
        self.call("is_Exposure", self._camID, IS_EXPOSURE_CMD_SET_EXPOSURE, ptr(ctypes.c_double(self.expmin)), ctypes.sizeof(ctypes.c_double()))
        self.call("is_SetDisplayMode", self._camID, IS_SET_DM_DIB)

        self.create_buffer()


    # close connection and release memory!
    def disconnect(self):
        """Disconnect a currently connected camera.
        """
        self.call("is_ExitCamera", self._camID)


    def stop(self):
        """Same as `disconnect`.

        .. versionadded:: 01-07-2016
        """
        self.disconnect()


    # sensor info
    def get_sensor_size(self):
        """Returns the sensor size as tuple: (width, height).

        If not connected yet, it returns a zero tuple.

        .. versionadded:: 01-07-2016
        """
        return self._swith, self._sheight


    # set hardware gain (0..100)
    def set_gain(self, gain):
        """Set the hardware gain.

        :param int gain: New gain setting (0 - 100).
        """
        self.call("is_SetHardwareGain", self._camID, max(0, min(int(gain), 100)), IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER)


    # returns gain
    def get_gain(self):
        """Returns current gain setting.
        """
        pParam = self.query("is_SetHardwareGain", self._camID, IS_GET_MASTER_GAIN, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER)
        return pParam


    def get_gain_limits(self):
        """Returns gain limits (*min, max, increment*).
        """
        return 0, 100, 1


    # switch gain boost on/ off
    def set_gain_boost(self, onoff):
        """Switch gain boost on or off.
        """
        if onoff:
            self.call("is_SetGainBoost", self._camID, IS_SET_GAINBOOST_ON)
        else:
            self.call("is_SetGainBoost", self._camID, IS_SET_GAINBOOST_OFF)


    # set blacklevel compensation
    def set_blacklevel(self, blck):
        """Set blacklevel compensation on or off.
        """
        nMode = ctypes.c_int(blck)
        self.call("is_Blacklevel", self._camID, IS_BLACKLEVEL_CMD_SET_MODE, ptr(nMode), ctypes.sizeof(nMode))


    # sets exposure time in ms
    def set_exposure(self, exp):
        """Set exposure time in milliseconds.
        """
        pParam = ctypes.c_double(exp)
        self.call("is_Exposure", self._camID, IS_EXPOSURE_CMD_SET_EXPOSURE, ptr(pParam), ctypes.sizeof(pParam))


    # returns exposure time in ms
    def get_exposure(self):
        """Returns current exposure time in milliseconds.
        """
        pParam = ctypes.c_double()
        self.call("is_Exposure", self._camID, IS_EXPOSURE_CMD_GET_EXPOSURE, ptr(pParam), ctypes.sizeof(pParam))
        return pParam.value


    def get_exposure_limits(self):
        """Returns the supported limits for the exposure time (*min, max, increment*).
        """
        return self.expmin, self.expmax, self.expinc


    # create image buffers
    def create_buffer(self):
        """Create image buffer for raw data from camera.

        .. note:: This function is automatically invoked by :py:func:`connect`.
        """
        # allocate memory for raw data from camera
        if self._image:
            self.call("is_FreeImageMem", self._camID, self._image, self._imgID)
        self._image = ctypes.c_char_p()
        self._imgID = ctypes.c_int()
        self.call("is_AllocImageMem", self._camID, self._swidth, self._sheight, self._bitsperpixel, ptr(self._image), ptr(self._imgID))
        self.call("is_SetImageMem", self._camID, self._image, self._imgID)


    # copy data from camera buffer to numpy frame buffer and return typecast to float
    def get_buffer(self):
        """Copy data from camera buffer to numpy array and return typecast to uint8.

        .. note:: This function is internally used by :py:func:`acquire`, :py:func:`acquireBinned`, and :py:func:`acquireMax` and there is normally no reason to directly call it.
        """
        # create usable numpy array for frame data
        if(self._bitsperpixel == 8):
            _framedata = np.zeros((self._sheight, self._swidth), dtype=np.uint8)
        else:
            _framedata = np.zeros((self._sheight, self._swidth, 3), dtype=np.uint8)

        self.call("is_CopyImageMem", self._camID, self._image, self._imgID, _framedata.ctypes.data_as(ctypes.c_char_p))
        return _framedata


    # captures N frames and returns the averaged image
    def acquire(self, N=1, native=False):
        """Synchronously captures some frames from the camera using the current settings and returns the averaged image.

        :param int N: Number of frames to acquire (> 1).
        :returns: Averaged image.
        """
        logger.debug(("acquire %d frames" % N))
        if not self._image:
            logger.debug("  create buffer..")
            self.create_buffer()

        data = None
        for i in range(int(N)):
            logger.debug("  wait for data..")
            ret = self.query("is_FreezeVideo", self._camID, IS_WAIT)
            while ret != IS_SUCCESS:
                logger.debug("is_FreezeVideo returned: {}".format(ret))
                time.sleep(0.1)
            logger.debug("  read data..")
            if data is None:
                if native:
                    data = self.get_buffer()
                else:
                    data = self.get_buffer().astype(float)
            else:
                data = data + self.get_buffer()
        if not native:
            data = data / float(N)

        return data


    # captures N frames and returns the fully binned arrays
    # along x and y directions and the maximum intensity in the array
    def acquireBinned(self, N=1):
        """Record N frames from the camera using the current settings and return fully binned 1d arrays averaged over the N frames.

        :param int N: Number of images to acquire.
        :returns: - Averaged 1d array fully binned over the x-axis.
                  - Averaged 1d array fully binned over the y-axis.
                  - Maximum pixel intensity before binning, e.g. to detect over illumination.
        """
        data = self.acquire(N)
        return np.sum(data, axis=0), np.sum(data, axis=1), np.amax(data)


    # returns the column / row with the maximum intensity
    def acquireMax(self, N=1):
        """Record N frames from the camera using the current settings and return the column / row with the maximum intensity.

        :param int N: Number of images to acquire.
        :returns: - Column with maximum intensity (1d array).
                  - Row with maximum intensity (1d array).
        """
        data = self.acquire(N)
        return data[np.argmax(np.data, axis=0), :], data[:, np.argmax(np.data, axis=1)]


# if __name__ == "__main__":

#     import pylab as pl
#     cam = uc480()
#     cam.connect()
#     img = cam.acquire(1)
#     pl.plot(np.mean(img, axis=0))
#     cam.disconnect()
#     pl.show()
