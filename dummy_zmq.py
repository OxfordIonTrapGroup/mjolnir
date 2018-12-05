#!/usr/bin/env python3.5
import argparse
import sys
import time
import zmq
import numpy as np
import itertools
import logging
from threading import Thread

from artiq.protocols.pc_rpc import simple_server_loop
from artiq.tools import verbosity_args, simple_network_args, init_logger
from image_test import generate_image

logger = logging.getLogger(__name__)

class Dummy:
    def __init__(self):
        space = np.linspace(0, 2*np.pi, num=10)
        means = np.array([200 + 100*np.sin(space), 400 + 200*np.cos(space)])
        frames = [generate_image(m)[2] for m in means.T]
        self._framegen = itertools.cycle(frames)
        self._frame = None
        self.quit = False
        self.dead = False
        self._frame_call_list = []
        self.acquisition_enabled = False
        t = Thread(target=self._acquisition_thread, daemon=True)
        t.start()

    def _acquisition_thread(self):
        symbols = itertools.cycle(r"\|/-")
        last_update = time.time()
        fps = None
        while self.quit is False:
            if self.acquisition_enabled:
                im = next(self._framegen)
                for f in self._frame_call_list:
                    f(im)
                self._frame = im

                now = time.time()
                dt = now - last_update
                last_update = now
                if fps is None and dt != 0:
                    fps = 1.0 / dt
                elif fps is None:
                    fps = 0.
                else:
                    s = np.clip(dt*3., 0, 1)
                    fps = fps * (1-s) + (1.0/dt) * s

                print(format("\r{} {:.1f} fps".format(next(symbols), fps), '<16'),
                    end='', flush=True)
            time.sleep(0.1)
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

    def register_callback(self, f):
        self._frame_call_list.append(f)

    def deregister_callback(self, f):
        if f in self._frame_call_list:
            self._frame_call_list.remove(f)

    def start_acquisition(self, single=False):
        """Turn on auto acquire"""
        def acquire_single_cb(_):
            """image is passed to callback and not used"""
            self.stop_acquisition()
            self.deregister_callback(acquire_single_cb)

        if single:
            self.register_callback(acquire_single_cb)
        self.counter = 0
        self.acquisition_enabled = True

    def single_acquisition(self):
        self.start_acquisition(single=True)

    def stop_acquisition(self):
        """Turn off auto acquire"""
        self.acquisition_enabled = False

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

    dev = Dummy()

    if args.broadcast_images:
        socket = create_zmq_server(args.zmq_bind, args.zmq_port)
        dev.register_callback(lambda im: socket.send_pyobj(im))

    try:
        simple_server_loop({"camera": dev}, args.bind, args.port)
    finally:
        dev.close()



if __name__ == "__main__":
    main()
