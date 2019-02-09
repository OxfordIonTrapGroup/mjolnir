#!/usr/bin/env python3.5
#
# Dummy implementation that closely matches the actual
# camera, just returns cached frames
#
# Also can be used a network server of dummy images
#
import time
import numpy as np
import itertools
import logging
from threading import Thread

from artiq.protocols.pc_rpc import simple_server_loop
from artiq.tools import init_logger

from mjolnir.frontend.server import get_argparser, run_server
from mjolnir.test.image_test import generate_image

logger = logging.getLogger(__name__)

class DummyCamera:
    def __init__(self, *args, **kwargs):
        # Cache some images to spew out
        num = 10
        space = np.linspace(0, 2*np.pi, num=num, endpoint=False)
        centroids = np.array([500 + 10*np.sin(space), 500 + 20*np.cos(space)])
        covs = np.outer(np.geomspace(1, 1e4, num=num), [1, 0, 0, 1.1]).reshape(-1, 2, 2)
        frames = [generate_image(c, cov=cov, noise=5) for c, cov in zip(centroids.T, covs)]

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

                # print("\r{} {: >4.1f} fps".format(
                #       next(symbols), fps), end='', flush=True)
            time.sleep(0.1)
        self.dead = True

    def ping(self):
        return True

    def close(self):
        self.quit = True
        while not self.dead:
            time.sleep(0.1)

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


def main():
    args = get_argparser().parse_args()
    init_logger(args)

    dev = DummyCamera()
    run_server(dev, args)


if __name__ == "__main__":
    main()
