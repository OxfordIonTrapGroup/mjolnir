#!/usr/bin/env python3.5
import asyncio
import argparse
import sys
import zmq
import numpy as np
import itertools
import logging
from quamash import QEventLoop
from PyQt5 import QtCore
from artiq.protocols.pc_rpc import simple_server_loop
from artiq.tools import verbosity_args, simple_network_args, init_logger

from mjolnir.test.image_test import generate_image
from mjolnir.tools import tools

logger = logging.getLogger(__name__)

class Dummy(QtCore.QObject):
    new_image = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()

        space = np.linspace(0, 2*np.pi, num=10, endpoint=False)
        centroids = np.array([200 + 100*np.sin(space), 400 + 200*np.cos(space)])
        frames = [generate_image(c) for c in centroids.T]

        # generate smaller images to see if calculation is limiting factor
        # (it is by far the limiting factor)
        # centroids = np.array([20 + 10*np.sin(space), 20+0*np.cos(space)])
        # frames = [generate(centroid=c, cov=[[4,0],[0,4]], m=50, n=50)
        #           for c in centroids.T]

        self._framegen = itertools.cycle(frames)
        self._frame = None
        self.acquisition_enabled = False
        self.symbols = itertools.cycle(r"\|/-")

        self._fps = None
        self._last_update = QtCore.QTime.currentTime()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._acquire)
        self.quit = False
        self.dead = False

    def _acquire(self):
        im = next(self._framegen)
        self.new_image.emit(im)

        self._frame = im
        self._last_update, self._fps = tools.update_rate(
            self._last_update, self._fps)

        print(format("\r{} {:.1f} fps".format(
            next(self.symbols), self._fps), '<16'),
            end='', flush=True)

        if self.quit:
            self.dead = True

    def ping(self):
        return True

    def close(self):
        self.quit = True
        while not self.dead:
            time.sleep(0.01)

    def get_image(self):
        return self._frame

    def get_exposure_params(self):
        return 0.9, 0.1, 30, 0.1

    def start_acquisition(self, single=False):
        """Turn on auto acquire"""
        if single:
            self._acquire()
        else:
            self.timer.start(100)

    def single_acquisition(self):
        self.start_acquisition(single=True)

    def stop_acquisition(self):
        """Turn off auto acquire"""
        self.timer.stop()

    def set_exposure_time(self, *args, **kwargs):
        pass

    def set_exposure_ms(self, *args, **kwargs):
        pass


def get_argparser():
    parser = argparse.ArgumentParser()
    simple_network_args(parser, 4000)
    verbosity_args(parser)
    parser.add_argument("--broadcast-images", action="store_true")
    parser.add_argument("--zmq-bind", default="*")
    parser.add_argument("--zmq-port", default=5555, type=int)
    return parser

def create_zmq_server(bind="*", port=5555):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.set_hwm(1)
    socket.bind("tcp://{}:{}".format(bind, port))
    return socket

def main():
    args = get_argparser().parse_args()
    init_logger(args)

    app = QtCore.QCoreApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    dev = Dummy()

    if args.broadcast_images:
        socket = create_zmq_server(args.zmq_bind, args.zmq_port)
        dev.new_image.connect(socket.send_pyobj)

    try:
        simple_server_loop({"camera": dev}, args.bind, args.port)
    finally:
        dev.close()
        loop.close()



if __name__ == "__main__":
    main()
