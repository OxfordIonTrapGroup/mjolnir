import asyncio
import itertools
import time
import atexit
import zmq
import zmq.asyncio
import pyqtgraph as pg
import numpy as np
import sys
import argparse
from PyQt5 import QtGui, QtWidgets, QtCore
from threading import Thread
from quamash import QEventLoop

from artiq.protocols.pc_rpc import Client
from new_image_tools import GaussianBeam



"""
Should abstract away the camera device and zmq reception:
that way we can have one subclass with an ARTIQ controller and zmq socket,
and another that just interfaces the camera directly for local operation
"""
class BeamDisplay:
    def __init__(self, loop, server, ctl_port, zmq_port):
        self.loop = loop
        self.ctl = Client(server, ctl_port)

        self._server = server
        self._ctl_port = ctl_port
        self._zmq_port = zmq_port
        self._processing = False
        self.fps = None
        self.last_update = pg.ptime.time()
        self.symbols = itertools.cycle(r"\|/-") # unicode didnt work(r"⠇⠋⠙⠸⠴⠦")

        self.init_ui()

        # asyncio operation
        # self._ctx = zmq.asyncio.Context()

        # threaded operation
        # self._ctx = zmq.Context()
        # t = Thread(
        #    target=self._recv_task,
        #    daemon=True)
        # t.start()

        # Qt timers (refresh rate is locked)
        qt_update = self.qt_update_factory()
        timer = QtCore.QTimer(self.win)
        timer.timeout.connect(qt_update)
        timer.start(50) # timeout ms

    def zmq_setup(self, ctx):
        sock = ctx.socket(zmq.SUB)
        sock.set_hwm(1)
        sock.connect("tcp://{}:{}".format(self._server, self._zmq_port))
        sock.setsockopt_string(zmq.SUBSCRIBE, '')
        sock.setsockopt(zmq.CONFLATE, 1)
        return sock

    def qt_update_factory(self):
        ctx = zmq.Context()
        sock = self.zmq_setup(ctx)
        sock.setsockopt(zmq.RCVTIMEO, 1)
        def qt_update():
            try:
                im = sock.recv_pyobj()
            except zmq.error.Again as e:
                pass
            else:
                self.update(im)
        return qt_update

    def _recv_task(self):
        ctx = zmq.Context()
        sock = self.zmq_setup(ctx)
        # shouldn't have a while True in this kind of thread
        while True:
            im = sock.recv_pyobj()
            self.update(im)

    async def recv_and_process(self):
        ctx = zmq.asyncio.Context()
        sock = self.zmq_setup(ctx)
        while True:
            im = await sock.recv_pyobj()
            self.update(im)

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

            im_fit = GaussianBeam.f(pxcrop, p)
            im_residuals = im_crop - im_fit

            self.zoom.setImage(im_crop, autoRange=False, autoLevels=False)
            # self.fit.setImage(im_fit, autoRange=False, autoLevels=False)
            self.residuals.setImage(im_residuals, autoRange=False, autoLevels=False)

            # just in case max pixel is not exactly centred
            px_x0 = np.unravel_index(np.argmax(im_fit), im_fit.shape)
            # x slice is horizontal
            self.x_slice.setData(im_crop[:,px_x0[1]], pxcrop[0,:,0])
            self.x_fit.setData(im_fit[:,px_x0[1]], pxcrop[0,:,0])

            self.y_slice.setData(im_crop[px_x0[0],:], pxcrop[1,0,:])
            self.y_fit.setData(im_fit[px_x0[0],:], pxcrop[1,0,:])

            self.fit_v_line.setValue(p['x0'][0])
            self.fit_h_line.setValue(p['x0'][1])

            w, v = np.linalg.eig(p['cov'])
            maj = np.argmax(w)
            min_ = maj - 1     # trick: we can always index with -1 or 0
            maj_angle = np.rad2deg(np.arctan2(v[1, maj], v[0, maj]))
            min_angle = np.rad2deg(np.arctan2(v[1, min_], v[0, min_]))
            self.fit_maj_line.setAngle(maj_angle)
            self.fit_min_line.setAngle(min_angle)
            centre = QtCore.QPointF(*(p['x0']-pxcrop[:,0,0]))
            self.fit_maj_line.setValue(centre)
            self.fit_maj_line.setValue(centre)

            self.isocurve.setLevel(np.amax(im_fit) / np.exp(2))
            self.isocurve.setData(im_fit)

        now = pg.ptime.time()
        dt = now - self.last_update
        self.last_update = now
        if self.fps is None:
            self.fps = 1.0 / dt
        else:
            s = np.clip(dt*3., 0, 1)
            self.fps = self.fps * (1-s) + (1.0/dt) * s

        print(format("\r{} {:.1f} fps".format(next(self.symbols), self.fps), '<16'),
            end='', flush=True)

        # QApplication.processEvents()
        self._processing = False

    def init_ui(self):
        self.win = QtWidgets.QMainWindow()
        self.widget = QtWidgets.QWidget(self.win)
        self.layout = QtWidgets.QHBoxLayout(self.win)

        # control panel
        self.info_pane = QtWidgets.QWidget(self.win)
        self.info_pane_layout = QtWidgets.QVBoxLayout(self.win)

        self._single = QtGui.QPushButton("Single Acquisition")
        self._start = QtGui.QPushButton("Start Acquisition")
        self._stop = QtGui.QPushButton("Stop Acquisition")
        self._single.clicked.connect(lambda: self.ctl.single_acquisition())
        self._start.clicked.connect(lambda: self.ctl.start_acquisition())
        self._stop.clicked.connect(lambda: self.ctl.stop_acquisition())

        self._exposure = QtGui.QDoubleSpinBox()
        self._exposure.setSuffix(" ms")
        self._get_exposure_params()
        # connect after finding params so we don't send accidental update
        self._exposure.valueChanged.connect(self._exposure_cb)

        self.info_pane_layout.setAlignment(QtCore.Qt.AlignTop)
        self.info_pane_layout.addWidget(self._start)
        self.info_pane_layout.addWidget(self._single)
        self.info_pane_layout.addWidget(self._stop)
        self.info_pane_layout.addWidget(self._exposure)
        self.info_pane.setLayout(self.info_pane_layout)

        self.init_graphics()

        # general
        self.layout.addWidget(self.info_pane)
        self.widget.setLayout(self.layout)
        self.win.setCentralWidget(self.widget)
        self.win.setGeometry(300, 300, 900, 600)


    def init_graphics(self):
        # images
        img = np.zeros((2,2))
        self.image = pg.ImageItem(img)
        self.zoom = pg.ImageItem(img)
        self.fit = pg.ImageItem(img)
        self.residuals = pg.ImageItem(img)
        self.x_fit = pg.PlotDataItem(np.zeros(2), pen={'width':2})
        self.x_slice = pg.PlotDataItem(np.zeros(2),pen=None, symbol='o', pxMode=True, symbolSize=4)
        self.y_fit = pg.PlotDataItem(np.zeros(2), pen={'width':2})
        self.y_slice = pg.PlotDataItem(np.zeros(2),pen=None, symbol='o', pxMode=True, symbolSize=4)

        self.g_layout = pg.GraphicsLayoutWidget(border=(80, 80, 80))

        options = {"lockAspect":True, "invertY":True}
        self.vb_image = self.g_layout.addViewBox(row=0, col=0, rowspan=4, **options)
        self.vb_zoom = self.g_layout.addViewBox(row=0, col=1, **options)
        # self.vb_fit = self.g_layout.addViewBox(row=1, col=1, **options)
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
        # self.vb_fit.addItem(self.fit)
        self.vb_residuals.addItem(self.residuals)
        self.vb_x.addItem(self.x_slice)
        self.vb_x.addItem(self.x_fit)
        self.vb_y.addItem(self.y_slice)
        self.vb_y.addItem(self.y_fit)

        self.vb_image.setRange(QtCore.QRectF(0, 0, 1280, 1280))
        self.vb_zoom.setRange(QtCore.QRectF(0, 0, 50, 50))
        # self.vb_fit.setRange(QtCore.QRectF(0, 0, 50, 50))
        self.vb_residuals.setRange(QtCore.QRectF(0, 0, 50, 50))
        self.vb_x.setRange(xRange=(0,255))
        self.vb_y.setRange(xRange=(0,255))
        self.vb_x.disableAutoRange(axis=self.vb_x.XAxis)
        self.vb_y.disableAutoRange(axis=self.vb_y.XAxis)

        self.g_layout.ci.layout.setColumnStretchFactor(0, 2)
        self.g_layout.ci.layout.setColumnStretchFactor(1, 1)

        self.layout.addWidget(self.g_layout, stretch=2)

    def start(self):
        self.win.show()
        try:
            #self.loop.create_task(self.recv_and_process())
            self.loop.run_forever()
        finally:
            self.close()

    def _get_exposure_params(self):
        val, min_, max_, step = self.ctl.get_exposure_params()
        self._exposure.setRange(min_, max_)
        self._exposure.setSingleStep(step)
        self._exposure.setValue(val)

    def _exposure_cb(self):
        exp = self._exposure.value()
        self.ctl.set_exposure_ms(exp)

    def _aoi_cb(self):
        pass

    def close(self):
        self.loop.close()
        self.ctl.close_rpc()
        self.ctl = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", "-s", type=str, required=True)
    parser.add_argument("--ctl-port", "-p", type=int, default=4000)
    parser.add_argument("--zmq-port", "-z", type=int, default=5555)

    args = parser.parse_args()

    global app
    app = QtWidgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    ctx = zmq.asyncio.Context()

    b = BeamDisplay(loop, args.server, args.ctl_port, args.zmq_port)
    b.start()


if __name__ == "__main__":
    main()
