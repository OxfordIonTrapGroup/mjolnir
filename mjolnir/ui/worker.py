import numpy as np
from PyQt5 import QtCore

from mjolnir.tools.image import GaussianBeam
from mjolnir.tools import tools


class Worker(QtCore.QObject):
    new_update = QtCore.pyqtSignal()

    def __init__(self, imageq, updateq):
        super().__init__()

        self._cps = None
        self._last_update = QtCore.QTime.currentTime()
        self._region = 50
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

        if (np.amax(im) - np.amin(im)) < 100:
            update = {'im': im, 'failure': "Contrast too low"}
            finish(update)
            return

        try:
            p = GaussianBeam.lsq_cropped(pxmap, im, region)
        except Exception as e:
            # We really don't want a fit failure to kill the GUI
            update = {'im': im, 'failure': str(e)}
            finish(update)
            return

        pxcrop, im_crop = GaussianBeam.crop(pxmap, im, p['x0'], region)
        if not len(pxcrop):
            # Centre was outside the actual image
            update = {'im': im, 'failure': "Bad centre"}
            finish(update)
            return

        im_fit = GaussianBeam.f(pxcrop, p)

        # residuals normalised to the pixel value
        im_err = np.sqrt(im_crop)
        im_err[im_err==0] = 1.0
        # Truncate because pixel values are integer, so for zero pixel value
        # our residual would otherwise look much worse than it is
        im_res = np.trunc(im_crop - im_fit)/im_err

        zoom_origin = pxcrop[:,0,0]
        # just in case max pixel is not exactly centred
        px_x0 = np.unravel_index(np.argmax(im_fit), im_fit.shape)
        px_x0 += zoom_origin

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
        zoom_centre = QtCore.QPointF(*(p['x0']-zoom_origin))

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
            'zoom_origin': zoom_origin,
            'zoom_centre': zoom_centre,
            'iso_level': iso_level
        }

        # add all the fit parameters to the update
        update.update(p)

        finish(update)


    @QtCore.pyqtSlot(int)
    def set_region(self, value):
        self._region = value
