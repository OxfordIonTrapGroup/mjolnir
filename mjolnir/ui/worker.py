import logging
import numpy as np
from PyQt5 import QtCore

from mjolnir.tools.image import GaussianBeam, auto_crop
from mjolnir.tools import tools


logger = logging.getLogger(__name__)


class Worker(QtCore.QObject):
    new_update = QtCore.pyqtSignal()

    def __init__(self, imageq, updateq):
        super().__init__()

        self._cps = None
        self._last_update = QtCore.QTime.currentTime()
        self._region = 20
        self._dwnsmp = None
        self.imageq = imageq
        self.updateq = updateq

    @QtCore.pyqtSlot()
    def process_imageq(self):
        try:
            im = self.imageq.popleft()
        except IndexError:
            return
        else:
            self.process_image(im)

    def process_image(self, im):
        def finish(update):
            # calulate number of updates per second
            self._last_update, self._cps = tools.update_rate(
                self._last_update, self._cps)
            update['cps'] = self._cps

            self.updateq.append(update)
            self.new_update.emit()

        m, n = im.shape
        pxmap = np.mgrid[0:m,0:n]
        region = self._region
        dwnsmp = self._dwnsmp

        if (np.amax(im) - np.amin(im)) < 100:
            update = {'im': im, 'failure': "Contrast too low"}
            finish(update)
            return

        try:
            im_crop, px_crop = auto_crop(im, dwnsmp_size=20)
            dwnsmp = px_crop[0,1,0] - px_crop[0,0,0]
        except Exception as e:
            logger.exception("auto_crop failed")
            update = {'im': im, 'failure': str(e)}
            finish(update)
            return

        try:
            p = GaussianBeam.fit(im_crop, pxmap=px_crop)
        except RuntimeError:
            update = {
                'im': im,
                'im_crop': im_crop,
                'failure': "Least squares fit failed"
            }
            finish(update)
            return
        except Exception as e:
            # We really don't want a fit failure to kill the GUI
            update = {'im': im, 'failure': str(e)}
            logger.exception("Fit failure")
            finish(update)
            return

        if np.any(p['x0'] < pxmap[:,0,0]) or np.any(p['x0'] > pxmap[:,-1,-1]):
            update = {'im': im, 'failure': "Centre outside image"}
            finish(update)
            return

        # Need to correct centre for downsampling
        zoom_centre = (p['x0'] - px_crop[:,0,0]) / dwnsmp + [0.5, 0.5]
        zoom_centre = QtCore.QPointF(*zoom_centre)

        im_fit = GaussianBeam.f(px_crop, p)

        # residuals normalised to the pixel value
        im_err = np.sqrt(im_crop)
        im_err[im_err==0] = 1.0
        # Truncate because pixel values are integer, so for zero pixel value
        # our residual would otherwise look much worse than it is
        im_res = np.trunc(im_crop - im_fit)/im_err

        px_x0 = np.around(p['x0']).astype(int)

        x = pxmap[0,:,0]
        x_slice = im[:,px_x0[1]]
        x_fit = GaussianBeam.f(pxmap[:,:,px_x0[1]],p)

        y = pxmap[1,0,:]
        y_slice = im[px_x0[0],:]
        y_fit = GaussianBeam.f(pxmap[:,px_x0[0],:],p)

        iso_level = np.amax(im_fit) / np.exp(2)

        # Correct for the fact that pixels are plotted with their origin at
        # the top left corner
        # Note that this comes after all fits have been calculated!
        p['x0'] += [0.5, 0.5]

        # construct our update dictionary
        update = {
            'im': im,
            'im_crop': im_crop,
            'im_fit': im_fit,
            'im_res': im_res,
            'x': x,
            'x_slice': x_slice,
            'x_fit': x_fit,
            'y': y,
            'y_slice': y_slice,
            'y_fit': y_fit,
            # 'zoom_origin': zoom_origin,
            'zoom_centre': zoom_centre,
            'iso_level': iso_level
        }

        # add all the fit parameters to the update
        update.update(p)

        finish(update)

    @QtCore.pyqtSlot('double')
    def set_region(self, value):
        value = int(value)

        if value <= 20:
            self._region = 20
            self._dwnsmp = None
        elif value <= 50:
            self._region = value
            self._dwnsmp = None
        else:
            self._dwnsmp = value // 40
            self._region = self._dwnsmp * 40

        logger.info("Set region: {}".format(self._region))
        logger.info("Set downsampling: {}".format(self._dwnsmp))
