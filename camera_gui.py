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
from collections import deque

from artiq.protocols.pc_rpc import Client
from new_image_tools import GaussianBeam

from dummy_zmq import Dummy


class _Worker(QtCore.QObject):
    new_update = QtCore.pyqtSignal()

    def __init__(self, imageq, updateq):
        super().__init__()

        self._cps = None
        self._last_update = pg.ptime.time()
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
        residual_max = np.amax(np.abs(im_residuals))
        residual_scale = 2 * residual_max/255
        # currently residuals is on [-255.,255.] and also is float
        # need ints on [0,255]
        im_res = 128 + (im_residuals / 2)
        im_res = np.clip(im_res, 0, 255).astype(int)

        # just in case max pixel is not exactly centred
        px_x0 = np.unravel_index(np.argmax(im_fit), im_fit.shape)
        px_x0 += pxcrop[:,0,0]

        x = pxmap[0,:,0]
        x_slice = im[:,px_x0[1]]
        x_fit = GaussianBeam.f(pxmap[:,:,px_x0[1]],p)

        y = pxmap[1,0,:]
        y_slice = im[px_x0[0],:]
        y_fit = GaussianBeam.f(pxmap[:,px_x0[0],:],p)

        # I think sub-pixel position is allowed?
        centre = QtCore.QPointF(*(p['x0']-pxcrop[:,0,0]))

        iso_level = np.amax(im_fit) / np.exp(2)

        update = {
            'im': im,
            'im_crop': im_crop,
            'im_fit': im_fit,
            'im_res': im_res,
            'residual_max': residual_max,
            'residual_scale': residual_scale,
            'x' : x,
            'x_slice': x_slice,
            'x_fit': x_fit,
            'y' : y,
            'y_slice': y_slice,
            'y_fit': y_fit,
            'centre': centre,
            'iso_level': iso_level
        }
        update.update(p)

        now = pg.ptime.time()
        dt = now - self._last_update
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


class BeamDisplay(QtWidgets.QMainWindow):
    new_image = QtCore.pyqtSignal()

    def __init__(self, camera):
        super().__init__()

        self.cam = camera
        # Pixel width in microns (get from camera?)
        self._px_width = 5.2

        # Deques discard the oldest value when full
        self.imageq = deque(maxlen=3)
        self.updateq = deque(maxlen=3)
        self.worker = _Worker(self.imageq, self.updateq)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)

        # must occur after moving worker!
        self.new_image.connect(self.worker.process_imageq)
        self.worker.new_update.connect(self.update)
        self.thread.start()

        self._fps = None
        self._last_update = pg.ptime.time()

        # difference between most positive and most negative
        # residuals in units of full scale
        self._residual_scale = 0.05

        self.init_ui()
        self.show()

    @QtCore.pyqtSlot()
    def queue_image(self, im):
        """Queues an image for fitting and plotting"""
        self.imageq.append(im)
        self.new_image.emit()

    @QtCore.pyqtSlot()
    def update(self):
        try:
            up = self.updateq.popleft()
        except IndexError:
            return

        options = {'autoRange': False, 'autoLevels': False}
        self.image.setImage(up['im'], **options)

        self.zoom.setImage(up['im_crop'], **options)
        self.residuals.setImage(up['im_res'], lut=self.residual_LUT, **options)
        self._residual_scale = up['residual_scale']
        self.res_label.setText("{:.1f}%".format(self._residual_scale*100))

        self.x_slice.setData(up['x'], up['x_slice'])
        self.x_fit.setData(up['x'], up['x_fit'])

        self.y_slice.setData(up['y_slice'], up['y'])
        self.y_fit.setData(up['y_fit'], up['y'])

        self.fit_v_line.setValue(up['x0'][0])
        self.fit_h_line.setValue(up['x0'][1])

        # I think sub-pixel position is allowed?
        self.fit_maj_line.setValue(up['centre'])
        self.fit_maj_line.setValue(up['centre'])
        self.fit_maj_line.setAngle(up['semimaj_angle'])
        self.fit_min_line.setAngle(up['semimin_angle'])

        self.isocurve.setLevel(up['iso_level'])
        self.isocurve.setData(up['im_fit'])

        def px_string(px):
            return "{:.1f}μm ({:.1f}px)".format(px*self._px_width, px)

        self.maj_radius.setText(px_string(up['semimaj']))
        self.min_radius.setText(px_string(up['semimin']))
        self.avg_radius.setText(px_string(up['avg_radius']))
        self.x_radius.setText(px_string(up['x_radius']))
        self.y_radius.setText(px_string(up['y_radius']))
        self.x_centroid.setText(px_string(up['x0'][0]))
        self.y_centroid.setText(px_string(up['x0'][1]))
        self.ellipticity.setText("{:.3f}".format(up['e']))
        self.residual_max.setText("{:.1f}".format(up['residual_max']))

        now = pg.ptime.time()
        dt = now - self._last_update
        self._last_update = now
        if self._fps is None:
            self._fps = 1.0 / dt
        else:
            s = np.clip(dt*3., 0, 1)
            self._fps = self._fps * (1-s) + (1.0/dt) * s

        self.fps.setText("{:.1f} fps".format(self._fps))
        self.cps.setText("{:.1f} cps".format(up['cps']))


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
        # Colour map for residuals is transparent when residual is zero
        colors = np.array([
            (0, 255, 255, 255),
            (0, 0, 255, 191),
            (0, 0, 0, 0),
            (255, 0, 0, 191),
            (255, 255, 0, 255)
        ], dtype=np.uint8)
        positions = [0.5 * (1 + 0.5 * scale * i) for i in range(-2,3)]
        self.residual_LUT = pg.ColorMap(positions, colors).getLookupTable(nPts=256)

    def rescale_LUT(self):
        self.residual_sf.setValue(self._residual_scale)

    def init_ui(self):
        self.widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout()

        self.init_graphics()
        self.init_info_pane()

        self.layout.addWidget(self.g_layout, stretch=2)
        self.layout.addWidget(self.info_pane)
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.setGeometry(300, 300, 1500, 600)

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

        self.maj_radius = QtGui.QLabel()
        self.min_radius = QtGui.QLabel()
        self.avg_radius = QtGui.QLabel()
        self.ellipticity = QtGui.QLabel()
        self.x_radius = QtGui.QLabel()
        self.y_radius = QtGui.QLabel()
        self.x_centroid = QtGui.QLabel()
        self.y_centroid = QtGui.QLabel()

        self.residual_max = QtGui.QLabel()
        self.residual_rescale = QtGui.QPushButton("Rescale")
        self.residual_rescale.clicked.connect(self.rescale_LUT)
        self.residual_sf = QtGui.QDoubleSpinBox()
        self.residual_sf.setRange(0.01, 1.)
        self.residual_sf.setSingleStep(0.05)
        self.residual_sf.valueChanged.connect(self.update_LUT)
        # deliberately afterwards to force update of lookup
        self.residual_sf.setValue(0.1)

        self.fps = QtGui.QLabel()
        self.cps = QtGui.QLabel()

        self.param_layout = QtWidgets.QFormLayout()
        self.param_layout.addRow(QtGui.QLabel("Beam Parameters"))
        self.param_layout.addRow(QtGui.QLabel("(all widths are 1/e^2)"))
        self.param_layout.addRow(QtGui.QWidget())
        self.param_layout.addRow("Semi-major radius:", self.maj_radius)
        self.param_layout.addRow("Semi-minor radius:", self.min_radius)
        self.param_layout.addRow("Average radius:", self.avg_radius)
        self.param_layout.addRow("Ellipticity:", self.ellipticity)
        self.param_layout.addRow(QtGui.QWidget())
        self.param_layout.addRow("X radius:", self.x_radius)
        self.param_layout.addRow("Y radius:", self.y_radius)
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
        self.info_pane_layout.addWidget(self.residual_rescale)
        self.info_pane_layout.addWidget(self.residual_sf)
        self.info_pane_layout.addStretch(3)
        self.info_pane_layout.addWidget(self.fps)
        self.info_pane_layout.addWidget(self.cps)

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
        self.vb_image = self.g_layout.addViewBox(row=0, col=0, rowspan=2, **options)
        self.vb_x = self.g_layout.addViewBox(row=2, col=0)
        self.vb_y = self.g_layout.addViewBox(row=0, col=1, rowspan=2)
        self.vb_zoom = self.g_layout.addViewBox(row=0, col=2, **options)
        self.vb_residuals = self.g_layout.addViewBox(row=1, col=2, **options)

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

        self.res_label = QtGui.QGraphicsSimpleTextItem(r"")
        self.res_label.setFont(QtGui.QFont("", 10))
        self.res_label.setBrush(pg.mkBrush((200,200,200)))
        self.res_label.setScale(0.2)
        # self.res_label.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations)

        self.vb_image.addItem(self.image)
        self.vb_image.addItem(self.fit_v_line)
        self.vb_image.addItem(self.fit_h_line)
        # Figure out how to overlay properly?
        # self.vb_image.addItem(self.x_slice)
        # self.vb_image.addItem(self.x_fit)
        # self.vb_image.addItem(self.y_slice)
        # self.vb_image.addItem(self.y_fit)
        self.vb_zoom.addItem(self.zoom)
        self.vb_zoom.addItem(self.fit_maj_line)
        self.vb_zoom.addItem(self.fit_min_line)
        self.vb_residuals.addItem(self.residuals)
        self.vb_residuals.addItem(self.res_label)
        self.vb_x.addItem(self.x_slice)
        self.vb_x.addItem(self.x_fit)
        self.vb_y.addItem(self.y_slice)
        self.vb_y.addItem(self.y_fit)

        self.vb_image.setRange(QtCore.QRectF(0, 0, 1280, 1024))
        self.vb_zoom.setRange(QtCore.QRectF(0, 0, 50, 50))
        self.vb_residuals.setRange(QtCore.QRectF(0, 0, 50, 50))

        self.vb_x.invertY(True)
        self.vb_y.invertY(True)
        self.vb_x.setXLink(self.vb_image)
        self.vb_y.setYLink(self.vb_image)
        self.vb_x.setRange(yRange=(0,255))
        self.vb_y.setRange(xRange=(0,255))
        self.vb_x.disableAutoRange(axis=self.vb_x.YAxis)
        self.vb_y.disableAutoRange(axis=self.vb_y.XAxis)

        self.g_layout.ci.layout.setColumnStretchFactor(0, 4)
        self.g_layout.ci.layout.setColumnStretchFactor(1, 1)
        self.g_layout.ci.layout.setColumnStretchFactor(2, 2)
        self.g_layout.ci.layout.setRowStretchFactor(0, 2)
        self.g_layout.ci.layout.setRowStretchFactor(1, 2)
        self.g_layout.ci.layout.setRowStretchFactor(2, 1)

        self.vb_x.setMinimumHeight(50)
        self.vb_y.setMinimumWidth(50)
        self.vb_x.setMaximumHeight(100)
        self.vb_y.setMaximumWidth(100)
        self.vb_image.setMinimumSize(640, 512)
        self.vb_zoom.setMinimumSize(320, 320)
        self.vb_residuals.setMinimumSize(320, 320)

        self.g_layout.setMinimumSize(1000,600)

    def add_tooltips(self):
        #TODO
        pass


def zmq_setup(ctx, server, port):
    sock = ctx.socket(zmq.SUB)
    sock.set_hwm(1)
    sock.connect("tcp://{}:{}".format(server, port))
    sock.setsockopt_string(zmq.SUBSCRIBE, '')
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 1)
    return sock


def remote(args):
    ### Remote operation ###
    camera = Client(args.server, args.artiq_port)
    b = BeamDisplay(camera)
    ctx = zmq.Context()
    sock = zmq_setup(ctx, args.server, args.zmq_port)

    def qt_update():
        try:
            im = sock.recv_pyobj()
        except zmq.error.Again as e:
            pass
        else:
            b.queue_image(im)

    timer = QtCore.QTimer(b)
    timer.timeout.connect(qt_update)
    timer.start(50) # timeout ms


def local(args):
    ### Local operation ###
    camera = Dummy()
    b = BeamDisplay(camera)
    camera.register_callback(lambda im: b.queue_image(im))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    remote_parser = subparsers.add_parser("remote",
        help="connect to a camera providing a ZMQ/Artiq network interface")
    remote_parser.add_argument("--server", "-s", type=str, required=True)
    remote_parser.add_argument("--artiq-port", "-p", type=int, default=4000)
    remote_parser.add_argument("--zmq-port", "-z", type=int, default=5555)
    remote_parser.set_defaults(func=remote)

    local_parser = subparsers.add_parser("local",
        help="connect to a local Thorlabs CMOS camera")
    local_parser.add_argument("--device", "-d", type=int, required=True,
        help="camera serial number")
    local_parser.set_defaults(func=local)

    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    args.func(args)

    sys.exit(app.exec_())


def test():
    app  = QtWidgets.QApplication(sys.argv)

    local()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
