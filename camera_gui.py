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
from pyqtgraph.widgets.RemoteGraphicsView import RemoteGraphicsView

from artiq.protocols.pc_rpc import Client
from new_image_tools_copy import GaussianBeam

class BeamDisplay:
    def __init__(self, loop, server, ctl_port, zmq_port):
        self.loop = loop
        self.ctl = Client(server, ctl_port)

        self._server = server
        self._ctl_port = ctl_port
        self._zmq_port = zmq_port
        self._ctx = zmq.Context()
        self._processing = False
        self.counter = 0
        self.last_update = 0
        self.symbols = itertools.cycle(r"\|/-")

        self.init_ui()
        t = Thread(
            target=self._recv_task,
            args=[self._ctx, self._server, self._zmq_port],
            daemon=True)
        t.start()

    def _recv_task(self, ctx, server, port):
        sock = self._ctx.socket(zmq.SUB)
        sock.set_hwm(1)
        sock.connect("tcp://{}:{}".format(server, port))
        sock.setsockopt_string(zmq.SUBSCRIBE, '')
        sock.setsockopt(zmq.CONFLATE, 1)
        while True:
            im = sock.recv_pyobj()
            self._process_task(im)

    def _process_task(self, im):
        if self._processing:
            return
        self._processing = True
        # self.image.setImage(im.T, autoRange=False, autoLevels=False)
        self.counter += 1

        if True:
            m, n = im.shape
            x = np.mgrid[0:m,0:n]
            p = GaussianBeam.two_step_MLE(x, im)
            xcrop, ycrop = GaussianBeam._crop(x, im, p['x0'])

            im_fit = GaussianBeam.f(xcrop, p)
            im_residuals = ycrop - im_fit

            #self.zoom.setImage(ycrop.T, autoRange=False, autoLevels=False)
            #self.residuals.setImage(im_residuals.T, autoRange=False, autoLevels=False)

        now = pg.ptime.time()
        fps = 1.0 / (now - self.last_update)
        self.last_update = now

        print(format("\r{} {:.1f} fps".format(next(self.symbols), fps), '<16'),
            end='', flush=True)

        self._processing = False

    async def recv_and_process(self):
        sock = self._ctx.socket(zmq.SUB)
        sock.set_hwm(1)
        sock.connect("tcp://{}:{}".format(self._server, self._zmq_port))
        sock.setsockopt_string(zmq.SUBSCRIBE, '')
        while True:
            im = await sock.recv_pyobj()
            self.image.setImage(im.T, autoRange=False, autoLevels=False)


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
        # connect after finding params so we don't send accidental
        self._exposure.valueChanged.connect(self._exposure_cb)

        self.info_pane_layout.setAlignment(QtCore.Qt.AlignTop)
        self.info_pane_layout.addWidget(self._start)
        self.info_pane_layout.addWidget(self._single)
        self.info_pane_layout.addWidget(self._stop)
        self.info_pane_layout.addWidget(self._exposure)
        self.info_pane.setLayout(self.info_pane_layout)

        #self.init_graphics()
        self.init_remote_graphics()

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
        self.residuals = pg.ImageItem(img)
        self.g_layout = pg.GraphicsLayoutWidget(border=(80, 80, 80))

        options = {"lockAspect":True, "invertY":True}
        self.vb_zoom = self.g_layout.addViewBox(row=0, col=1, **options)
        self.vb_residuals = self.g_layout.addViewBox(row=1, col=1, **options)
        self.vb_image = self.g_layout.addViewBox(row=0, col=0, rowspan=2, **options)

        self.vb_image.addItem(self.image)
        self.vb_zoom.addItem(self.zoom)
        self.vb_residuals.addItem(self.residuals)

        self.g_layout.ci.layout.setColumnStretchFactor(0, 2)
        self.g_layout.ci.layout.setColumnStretchFactor(1, 1)

        self.layout.addWidget(self.g_layout, stretch=2)

    def init_remote_graphics(self):
        v = RemoteGraphicsView(debug=False)
        v.show()
        plt = v.pg.PlotItem()

        img = np.random.randint(low=0, high=255, size=(1280,1024))
        self.image = v.pg.ImageView(view=plt)
        self.image.setImage(img)
        v.setCentralItem(plt)

        self.zoom = v.pg.ImageItem(img)
        self.residuals = v.pg.ImageItem(img)
        # self.g_layout = v.pg.GraphicsLayoutWidget(self.win, border=(80, 80, 80))

        # options = {"lockAspect":True, "invertY":True}
        # self.vb_zoom = self.g_layout.addViewBox(row=0, col=1, **options)
        # self.vb_residuals = self.g_layout.addViewBox(row=1, col=1, **options)
        # self.vb_image = self.g_layout.addViewBox(row=0, col=0, rowspan=2, **options)

        # self.vb_image.addItem(self.image)
        # self.vb_zoom.addItem(self.zoom)
        # self.vb_residuals.addItem(self.residuals)

        # self.g_layout.ci.layout.setColumnStretchFactor(0, 2)
        # self.g_layout.ci.layout.setColumnStretchFactor(1, 1)


    def start(self):
        self.win.show()
        try:
            # self.loop.create_task(self.recv_and_process())
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

    app = QtWidgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    ctx = zmq.asyncio.Context()

    b = BeamDisplay(loop, args.server, args.ctl_port, args.zmq_port)
    b.start()


if __name__ == "__main__":
    main()
