import numpy as np
from PyQt5 import QtCore
from new_image_tools import GaussianBeam


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
        m, n = im.shape
        pxmap = np.mgrid[0:m,0:n]

        region = self._region
        p = GaussianBeam.two_step_MLE(pxmap, im, region)
        pxcrop, im_crop = GaussianBeam.crop(pxmap, im, p['x0'], region)

        im_fit = GaussianBeam.f(pxcrop, p)
        im_residuals = im_crop - im_fit

        r_max = np.amax(np.abs(im_residuals))
        r_scale = 2 * r_max/255
        r_fraction = r_max/np.amax(im_fit)
        # currently residuals is on [-255.,255.] and also is float
        # need ints on [0,255]
        # autoscale to make best use of our colour map
        im_res = 127.5 + (im_residuals / r_scale)
        im_res = np.clip(im_res, 0, 255).astype(int)

        # legend for residuals
        nticks = 5
        legend = {"{:.1f}%".format(100*frac):val
                  for (frac, val) in zip(
                      np.linspace(-r_fraction, r_fraction, nticks),
                      np.linspace(0, 1, nticks))}

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

        # Subpixel position allowed but ignored
        zoom_centre = QtCore.QPointF(*(p['x0']-pxcrop[:,0,0]))

        iso_level = np.amax(im_fit) / np.exp(2)

        # construct our update dictionary
        update = {
            'im': im,
            'im_crop': im_crop,
            'im_fit': im_fit,
            'im_res': im_res,
            'legend': legend,
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

        # calulate number of updates per second
        now = QtCore.QTime.currentTime()
        dt = float(self._last_update.msecsTo(now))/1000
        self._last_update = now
        if self._cps is None:
            self._cps = 1.0 / dt
        else:
            s = np.clip(dt*3., 0, 1)
            self._cps = self._cps * (1-s) + (1.0/dt) * s

        update['cps'] = self._cps

        self.updateq.append(update)
        self.new_update.emit()
        self._working = False

    @QtCore.pyqtSlot(int)
    def set_region(self, value):
        self._region = value
