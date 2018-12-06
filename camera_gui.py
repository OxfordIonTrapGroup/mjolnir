import asyncio
import itertools
import zmq
import pyqtgraph as pg
import numpy as np
import sys
import argparse
import subprocess
from PyQt5 import QtGui, QtWidgets, QtCore
from quamash import QEventLoop

from artiq.protocols.pc_rpc import Client
from new_image_tools import GaussianBeam

from dummy_zmq import Dummy


class BeamDisplay(QtWidgets.QMainWindow):
    def __init__(self, camera):
        super().__init__()

        self.cam = camera

        # Pixel width in microns (get from camera?)
        self._px_width = 5.2

        self._processing = False
        self._fps = None
        self._last_update = pg.ptime.time()
        self._symbols = itertools.cycle(r"\|/-") # unicode didnt work(r"⠇⠋⠙⠸⠴⠦")

        self.init_ui()
        self.show()

    def update(self, im):
        if self._processing:
            return
        self._processing = True
        self.image.setImage(im, autoRange=False, autoLevels=False)

        if True:
            region = 50
            m, n = im.shape
            pxmap = np.mgrid[0:m,0:n]
            p = GaussianBeam.two_step_MLE(pxmap, im)
            pxcrop, im_crop = GaussianBeam.crop(pxmap, im, p['x0'], region)
            self.zoom.setImage(im_crop, autoRange=False, autoLevels=False)

            im_fit = GaussianBeam.f(pxcrop, p)
            im_residuals = im_crop - im_fit

            # currently residuals is on [-255.,255.] and also is float
            # need ints on [0,255]
            im_res = 127 + (im_residuals / 2)
            im_res = np.clip(im_res, 0, 255).astype(int)
            self.residuals.setImage(im_res,
                autoRange=False, autoLevels=False, lut=self.residual_LUT)

            # just in case max pixel is not exactly centred
            px_x0 = np.unravel_index(np.argmax(im_fit), im_fit.shape)
            # x slice is horizontal
            self.x_slice.setData(im_crop[:,px_x0[1]], pxcrop[0,:,0])
            self.x_fit.setData(im_fit[:,px_x0[1]], pxcrop[0,:,0])

            self.y_slice.setData(im_crop[px_x0[0],:], pxcrop[1,0,:])
            self.y_fit.setData(im_fit[px_x0[0],:], pxcrop[1,0,:])

            self.fit_v_line.setValue(p['x0'][0])
            self.fit_h_line.setValue(p['x0'][1])

            params = GaussianBeam.covariance_to_gaussian_params(p['cov'])

            centre = QtCore.QPointF(*(p['x0']-pxcrop[:,0,0]))
            self.fit_maj_line.setValue(centre)
            self.fit_maj_line.setValue(centre)
            self.fit_maj_line.setAngle(params['maj_angle'])
            self.fit_min_line.setAngle(params['min_angle'])

            self.isocurve.setLevel(np.amax(im_fit) / np.exp(2))
            self.isocurve.setData(im_fit)

            def px_string(px):
                return "{:.1f}μm ({:.1f}px)".format(px*self._px_width, px)

            self.maj_width.setText(px_string(params['maj_width']))
            self.min_width.setText(px_string(params['min_width']))
            self.x_width.setText(px_string(params['x_width']))
            self.y_width.setText(px_string(params['y_width']))
            self.x_centroid.setText(px_string(p['x0'][0]))
            self.y_centroid.setText(px_string(p['x0'][1]))
            self.ellipticity.setText("{:.3f}".format(params['e']))
            self.residual_max.setText("{:.1f}".format(np.amax(np.abs(im_residuals))))


        now = pg.ptime.time()
        dt = now - self._last_update
        self._last_update = now
        if self._fps is None:
            self._fps = 1.0 / dt
        else:
            s = np.clip(dt*3., 0, 1)
            self._fps = self._fps * (1-s) + (1.0/dt) * s

        self.fps.setText("{:.1f} fps".format(self._fps))

        # QApplication.processEvents()
        self._processing = False

    def _get_exposure_params(self):
        val, min_, max_, step = self.cam.get_exposure_params()
        self._exposure.setRange(min_, max_)
        self._exposure.setSingleStep(step)
        self._exposure.setValue(val)

    def _exposure_cb(self):
        exp = self._exposure.value()
        self.cam.set_exposure_ms(exp)

    def _aoi_cb(self):
        pass

    def update_LUT(self, scale):
        # have to loop twice to avoid reordering
        ticks = self.gradient.listTicks()
        for i in range(5):
            if i == 2:
                continue
            value = 0.5 * (1 + scale * (i-2) * 0.5)
            self.gradient.setTickValue(ticks[i][0], value)
        self.residual_LUT = self.gradient.getLookupTable(256)

    def init_ui(self):
        self.widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout()

        self.init_graphics()
        self.init_info_pane()

        self.layout.addWidget(self.g_layout, stretch=2)
        self.layout.addWidget(self.info_pane)
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.setGeometry(300, 300, 900, 600)

    def init_info_pane(self):
        self._single = QtGui.QPushButton("Single Acquisition")
        self._start = QtGui.QPushButton("Start Acquisition")
        self._stop = QtGui.QPushButton("Stop Acquisition")
        self._single.clicked.connect(lambda: self.cam.single_acquisition())
        self._start.clicked.connect(lambda: self.cam.start_acquisition())
        self._stop.clicked.connect(lambda: self.cam.stop_acquisition())

        self._exposure = QtGui.QDoubleSpinBox()
        self._exposure.setSuffix(" ms")
        self._get_exposure_params()
        # connect after finding params so we don't send accidental update
        self._exposure.valueChanged.connect(self._exposure_cb)

        self.maj_width = QtGui.QLabel()
        self.min_width = QtGui.QLabel()
        self.ellipticity = QtGui.QLabel()
        self.x_width = QtGui.QLabel()
        self.y_width = QtGui.QLabel()
        self.x_centroid = QtGui.QLabel()
        self.y_centroid = QtGui.QLabel()

        self.residual_max = QtGui.QLabel()
        self.gradient = pg.GradientWidget()
        self.gradient.loadPreset('bipolar')
        self.gradient.setTickColor(self.gradient.getTick(2),
            pg.mkColor(0,0,255,127))
        self.gradient.setTickColor(self.gradient.getTick(2),
            pg.mkColor(0,0,0,0))
        self.gradient.setTickColor(self.gradient.getTick(3),
            pg.mkColor(255,0,0,127))

        self.residual_sf = QtGui.QDoubleSpinBox()
        self.residual_sf.setRange(0.01, 1.)
        self.residual_sf.valueChanged.connect(self.update_LUT)
        # deliberately afterwards to force update of lookup
        self.residual_sf.setValue(1.)

        self.fps = QtGui.QLabel()

        self.param_layout = QtWidgets.QFormLayout()
        self.param_layout.addRow(QtGui.QLabel("Beam Parameters"))
        self.param_layout.addRow(QtGui.QLabel("(all widths are 1/e^2)"))
        self.param_layout.addRow(QtGui.QWidget())
        self.param_layout.addRow("Semi-major axis:", self.maj_width)
        self.param_layout.addRow("Semi-minor axis:", self.min_width)
        self.param_layout.addRow("Ellipticity:", self.ellipticity)
        self.param_layout.addRow(QtGui.QWidget())
        self.param_layout.addRow("X axis:", self.x_width)
        self.param_layout.addRow("Y axis:", self.y_width)
        self.param_layout.addRow(QtGui.QWidget())
        self.param_layout.addRow("X position:", self.x_centroid)
        self.param_layout.addRow("Y position:", self.y_centroid)
        self.param_layout.addRow(QtGui.QWidget())

        self.param_widget = QtGui.QWidget()
        self.param_widget.setLayout(self.param_layout)

        self.info_pane_layout = QtWidgets.QVBoxLayout()
        self.info_pane_layout.setAlignment(QtCore.Qt.AlignTop)
        self.info_pane_layout.addWidget(self._start)
        self.info_pane_layout.addWidget(self._single)
        self.info_pane_layout.addWidget(self._stop)
        self.info_pane_layout.addWidget(self._exposure)
        self.info_pane_layout.addStretch(1)
        self.info_pane_layout.addWidget(self.param_widget)
        self.info_pane_layout.addStretch(1)
        self.info_pane_layout.addWidget(self.residual_max)
        self.info_pane_layout.addWidget(self.residual_sf)
        self.info_pane_layout.addWidget(self.gradient)
        self.info_pane_layout.addStretch(3)
        self.info_pane_layout.addWidget(self.fps)

        self.info_pane = QtWidgets.QWidget(self)
        self.info_pane.setLayout(self.info_pane_layout)

    def init_graphics(self):
        # images
        img = np.zeros((2,2))
        self.image = pg.ImageItem(img)
        self.zoom = pg.ImageItem(img)
        self.residuals = pg.ImageItem(img)
        self.x_fit = pg.PlotDataItem(np.zeros(2), pen={'width':2})
        self.x_slice = pg.PlotDataItem(np.zeros(2), pen=None, symbol='o', pxMode=True, symbolSize=4)
        self.y_fit = pg.PlotDataItem(np.zeros(2), pen={'width':2})
        self.y_slice = pg.PlotDataItem(np.zeros(2), pen=None, symbol='o', pxMode=True, symbolSize=4)

        self.g_layout = pg.GraphicsLayoutWidget(border=(80, 80, 80))

        options = {"lockAspect":True, "invertY":True}
        self.vb_image = self.g_layout.addViewBox(row=0, col=0, rowspan=4, **options)
        self.vb_zoom = self.g_layout.addViewBox(row=0, col=1, **options)
        self.vb_residuals = self.g_layout.addViewBox(row=1, col=1, **options)
        self.vb_x = self.g_layout.addViewBox(row=2, col=1)
        self.vb_y = self.g_layout.addViewBox(row=3, col=1)

        color = pg.mkColor(40,40,40)
        self.vb_image.setBackgroundColor(color)
        self.vb_zoom.setBackgroundColor(color)
        self.vb_residuals.setBackgroundColor(color)
        self.vb_x.setBackgroundColor(color)
        self.vb_y.setBackgroundColor(color)
        self.g_layout.setBackground(color)

        fit_pen = pg.mkPen('y')
        self.fit_v_line = pg.InfiniteLine(pos=1, angle=90, pen=fit_pen)
        self.fit_h_line = pg.InfiniteLine(pos=1, angle=0, pen=fit_pen)
        zoom_centre = QtCore.QPointF(25,25)
        self.fit_maj_line = pg.InfiniteLine(pos=zoom_centre, pen=pg.mkPen('g'))
        self.fit_min_line = pg.InfiniteLine(pos=zoom_centre, pen=pg.mkPen('r'))
        self.isocurve = pg.IsocurveItem(pen=fit_pen)
        self.isocurve.setParentItem(self.zoom)

        self.vb_image.addItem(self.image)
        self.vb_image.addItem(self.fit_v_line)
        self.vb_image.addItem(self.fit_h_line)
        self.vb_zoom.addItem(self.zoom)
        self.vb_zoom.addItem(self.fit_maj_line)
        self.vb_zoom.addItem(self.fit_min_line)
        self.vb_residuals.addItem(self.residuals)
        self.vb_x.addItem(self.x_slice)
        self.vb_x.addItem(self.x_fit)
        self.vb_y.addItem(self.y_slice)
        self.vb_y.addItem(self.y_fit)

        self.vb_image.setRange(QtCore.QRectF(0, 0, 1280, 1280))
        self.vb_zoom.setRange(QtCore.QRectF(0, 0, 50, 50))
        self.vb_residuals.setRange(QtCore.QRectF(0, 0, 50, 50))
        self.vb_x.setRange(xRange=(0,255))
        self.vb_y.setRange(xRange=(0,255))
        self.vb_x.disableAutoRange(axis=self.vb_x.XAxis)
        self.vb_y.disableAutoRange(axis=self.vb_y.XAxis)

        self.g_layout.ci.layout.setColumnStretchFactor(0, 2)
        self.g_layout.ci.layout.setColumnStretchFactor(1, 1)



def zmq_setup(ctx, server, port):
    sock = ctx.socket(zmq.SUB)
    sock.set_hwm(1)
    sock.connect("tcp://{}:{}".format(server, port))
    sock.setsockopt_string(zmq.SUBSCRIBE, '')
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 1)
    return sock


def remote(server, ctl_port, zmq_port):
    ### Remote operation ###
    camera = Client(server, ctl_port)
    b = BeamDisplay(camera)
    ctx = zmq.Context()
    sock = zmq_setup(ctx, server, zmq_port)

    def qt_update():
        try:
            im = sock.recv_pyobj()
        except zmq.error.Again as e:
            pass
        else:
            b.update(im)

    timer = QtCore.QTimer(b)
    timer.timeout.connect(qt_update)
    timer.start(50) # timeout ms


def local():
    ### Local operation ###
    camera = Dummy()
    b = BeamDisplay(camera)
    camera.register_callback(lambda im: b.update(im))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", "-s", type=str, required=True)
    parser.add_argument("--ctl-port", "-p", type=int, default=4000)
    parser.add_argument("--zmq-port", "-z", type=int, default=5555)

    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    remote(args.server, args.ctl_port, args.zmq_port)

    sys.exit(app.exec_())



if __name__ == "__main__":
    main()
